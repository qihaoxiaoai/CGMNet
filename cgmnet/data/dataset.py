# cgmnet/data/dataset.py
import io
import torch
import lmdb
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import warnings

from cgmnet.utils.data_utils import get_task_config, get_split_indices
from cgmnet.data.featurizer import CGMNetFeaturizer
from cgmnet.utils.register import register_dataset
from tqdm import tqdm

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")


@register_dataset('cgmnet_pretrain')
class CGMNetPretrainDataset(Dataset):
    """
    预训练数据集：
      - knodes 使用 np.load(..., mmap_mode='r') 只读内存映射
      - __getitem__ 按索引读取一行 → 清洗 → torch.from_numpy(row.copy())
    """
    def __init__(self, lmdb_path: Path, descriptor_root: Path, knodes_to_load: List[str] = None, downsize: int = -1):
        self.lmdb_path_str = str(lmdb_path)
        self.db = None

        # 从 LMDB 读取样本数
        with lmdb.open(self.lmdb_path_str, readonly=True, lock=False) as env:
            with env.begin() as txn:
                self.num_examples = int(txn.get('num_examples'.encode()))
        if downsize > 0:
            self.num_examples = min(self.num_examples, downsize)

        # knodes 使用 memmap 并对齐长度
        self.knodes: Dict[str, np.memmap] = {}
        if knodes_to_load:
            print(f"Loading and sanitizing knowledge features for pre-training: {knodes_to_load}")
            effective_len = self.num_examples
            for name in knodes_to_load:
                fp_path = descriptor_root / f"{name}.npy"
                if fp_path.exists():
                    mm = np.load(fp_path, mmap_mode='r')
                    if mm.shape[0] < effective_len:
                        effective_len = mm.shape[0]
                    self.knodes[name] = mm
                else:
                    print(f"Warning: Knodes file not found: {fp_path}")
            self.num_examples = min(self.num_examples, effective_len)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        # 惰性打开 LMDB
        if self.db is None:
            self.db = lmdb.open(self.lmdb_path_str, readonly=True, lock=False, readahead=False)
        with self.db.begin() as txn:
            graph_bytes = txn.get(str(idx).encode())
        graph_dict = torch.load(io.BytesIO(graph_bytes), weights_only=False)

        # 逐行读取 knodes
        knodes_for_item = {}
        for name, mm in self.knodes.items():
            row = mm[idx]
            if row.dtype != np.float32:
                row = row.astype(np.float32, copy=False)
            row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
            knodes_for_item[name] = torch.from_numpy(row.copy())

        dummy_label = torch.tensor(-1.0)
        return graph_dict, dummy_label, knodes_for_item


@register_dataset('cgmnet_finetune')
class CGMNetFinetuneDataset(Dataset):
    """
    微调数据集：
      - 明确初始化 num_examples
      - knodes 与 split 对齐（memmap + index_map）
      - 可选：图缓存（LMDB），默认关闭；开启后首跑写入、后续直接读取
    """
    def __init__(
        self,
        dataset_name: str,
        dataset_root: Path,
        split: str,
        featurizer: CGMNetFeaturizer,
        knodes_to_load: List[str] = None,
        scaffold_id: int = 0,
        # === 新增（默认关闭，不影响旧行为） ===
        cache_graphs: bool = False,
        cache_dir: Path | None = None,
    ):
        dataset_dir = dataset_root / dataset_name
        csv_path = dataset_dir / f"{dataset_name}.csv"
        df_full = pd.read_csv(csv_path)

        # 读取 split
        try:
            split_indices = get_split_indices(dataset_dir / "splits", scaffold_id)
        except FileNotFoundError:
            print(f"Warning: Split file not found for {dataset_name}. Falling back to random 80/10/10 split.")
            indices = np.random.permutation(len(df_full))
            train_end, valid_end = int(0.8 * len(df_full)), int(0.9 * len(df_full))
            split_indices = {
                'train': indices[:train_end],
                'valid': indices[train_end:valid_end],
                'test': indices[valid_end:]
            }

        # 本 split 的原始索引映射（相对于完整 CSV）
        index_map = np.array(split_indices[split], dtype=np.int64)

        # 提前生成本 split 的 DataFrame 视图
        df = df_full.iloc[index_map].reset_index(drop=True)

        # 预置成员
        self.index_map = index_map
        self.knodes: Dict[str, np.ndarray] = {}
        self.smiles: List[str] = []
        self.targets = None
        self.graphs = None

        # 读取 knodes（memmap，不做子集拷贝），并保证 index_map 不越界
        feature_dir = dataset_dir / "features"
        if knodes_to_load:
            print(f"Loading and sanitizing knowledge features for fine-tuning: {knodes_to_load}")
            knode_lens = []
            for name in knodes_to_load:
                fp_path = feature_dir / f"{name}.npy"
                if not fp_path.exists():
                    raise FileNotFoundError(
                        f"Knodes file not found: {fp_path}. "
                        f"Run 02_generate_features.py for '{dataset_name}' first."
                    )
                mm = np.load(fp_path, mmap_mode='r')
                self.knodes[name] = mm
                knode_lens.append(mm.shape[0])

            if len(knode_lens) > 0:
                min_len = int(min(knode_lens))
                good_mask = (self.index_map < min_len)
                if not bool(np.all(good_mask)):
                    drop_cnt = int((~good_mask).sum())
                    print(f"Warning: {drop_cnt} indices in split '{split}' exceed available knode rows ({min_len}). They will be dropped.")
                    self.index_map = self.index_map[good_mask]
                    df = df.iloc[good_mask].reset_index(drop=True)

        # 初始化样本数（避免 AttributeError）
        self.num_examples = int(len(df))

        # 基本标签与 SMILES
        self.smiles = df["smiles"].tolist()
        self.targets = torch.tensor(df.drop("smiles", axis=1).values, dtype=torch.float32)

        # 任务配置与归一化统计（训练集才计算）
        task_config = get_task_config(dataset_name, dataset_root)
        if task_config['task_type'] == 'reg' and split == 'train':
            means, stds = [], []
            for i in range(self.targets.shape[1]):
                col = self.targets[:, i]
                mask = torch.isfinite(col)
                if mask.any():
                    v = col[mask]
                    means.append(torch.mean(v))
                    stds.append(torch.std(v) if v.numel() > 1 else torch.tensor(1.0))
                else:
                    means.append(torch.tensor(0.0))
                    stds.append(torch.tensor(1.0))
            self.target_mean = torch.stack(means)
            self.target_std = torch.stack(stds)
            self.target_std[self.target_std == 0] = 1.0

        # ===== 可选：图缓存（默认关闭，打开时写/读 LMDB） =====
        self._cache_enabled = bool(cache_graphs)
        self._cache_db_path = None
        self._cache_env = None
        self._featurizer = featurizer

        if self._cache_enabled:
            cache_root = Path(cache_dir) if cache_dir is not None else (dataset_dir / "graph_cache")
            cache_root.mkdir(parents=True, exist_ok=True)
            self._cache_db_path = cache_root / f"{dataset_name}.lmdb"

            # 写入缺失项（单进程顺序写，不影响 DataLoader 多进程读取）
            env = lmdb.open(str(self._cache_db_path), map_size=int(8 * 1024**3), subdir=True, readonly=False, lock=True)
            valid_mask = np.ones(len(self.smiles), dtype=bool)
            for i, s in enumerate(tqdm(self.smiles, desc=f"Featurizing+Caching[{split}]")):
                key = str(int(self.index_map[i])).encode()  # 用 CSV 行号做键
                with env.begin() as txn:
                    b = txn.get(key)
                if b is None:
                    g = self._featurizer(s)
                    if g is None:
                        valid_mask[i] = False
                        continue
                    buf = io.BytesIO()
                    torch.save(g, buf)
                    with env.begin(write=True) as txn:
                        txn.put(key, buf.getvalue())
            env.sync()
            env.close()

            # 剔除失败项并同步三元组（与原逻辑一致）
            if not bool(np.all(valid_mask)):
                num_failed = int((~valid_mask).sum())
                print(f"Warning: {num_failed} molecules failed to featurize and were removed.")
                self.index_map = self.index_map[valid_mask]
                self.smiles = [self.smiles[i] for i in range(len(valid_mask)) if valid_mask[i]]
                self.targets = self.targets[valid_mask]
            # 惰性只读打开（每个 worker 自己开句柄）
            self.graphs = None
        else:
            # 原始路径：直接内存构图（保持旧行为）
            print(f"Featurizing {len(self.smiles)} molecules for the '{split}' split...")
            self.graphs = [featurizer(s) for s in tqdm(self.smiles)]

            valid_indices = [i for i, g in enumerate(self.graphs) if g is not None]
            if len(valid_indices) < len(self.graphs):
                num_failed = len(self.graphs) - len(valid_indices)
                print(f"Warning: {num_failed} molecules failed to featurize and were removed.")
                self.graphs = [self.graphs[i] for i in valid_indices]
                self.targets = self.targets[valid_indices]
                self.smiles = [self.smiles[i] for i in valid_indices]
                self.index_map = self.index_map[valid_indices]

        # 最终样本数
        self.num_examples = int(len(self.smiles))

    def __len__(self):
        return self.num_examples

    def _open_cache_env(self):
        if (self._cache_env is None) and self._cache_enabled:
            # DataLoader 多进程下，每个 worker 会各自调用一次
            self._cache_env = lmdb.open(str(self._cache_db_path), readonly=True, lock=False, readahead=False)

    def __getitem__(self, idx):
        # knodes：按 index_map 映射到全量特征的行
        knodes_for_item = {}
        if self.knodes:
            base_idx = int(self.index_map[idx])
            for name, mm in self.knodes.items():
                row = mm[base_idx]
                if row.dtype != np.float32:
                    row = row.astype(np.float32, copy=False)
                row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
                knodes_for_item[name] = torch.from_numpy(row.copy())

        # 图：缓存启用→LMDB 读取；否则用内存列表
        if self._cache_enabled:
            self._open_cache_env()
            key = str(int(self.index_map[idx])).encode()
            with self._cache_env.begin() as txn:
                b = txn.get(key)
            if b is None:
                # 极端兜底（理论不会触发）：现算现用
                g = self._featurizer(self.smiles[idx])
            else:
                g = torch.load(io.BytesIO(b), weights_only=False)
        else:
            g = self.graphs[idx]

        return g, self.targets[idx], knodes_for_item, self.smiles[idx]

