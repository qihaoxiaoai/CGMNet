# cgmnet/data/featurizer.py
from typing import Dict, Any, Tuple, List, Optional

import torch
import dgl
import networkx as nx

from cgmnet.data.fragmentizer import (
    BaseFragmentizer,
    build_fragmentizer,
    FragmentMeta,
)
from cgmnet.data.frag_algos import BricsFragmentizer
from cgmnet.utils.chem_utils import smi_to_mol, atom_featurizer, bond_featurizer


class CGMNetFeaturizer:
    def __init__(
        self,
        vocab_path: str,
        order: Optional[int] = None,
        max_path_length: int = 5,
        fragmentizer: BaseFragmentizer | None = None,
        frag_method: str = "dove",
        overlap_degree: int | None = None,
    ):
        """
        Parameters
        ----------
        vocab_path : str
            词表路径。第一行 JSON 会被用来读取 meta（line_order / overlap_degree / kekulize / frag_method 等）。
        order : int | None
            == k_line，line graph 转换的阶数（主要给 DOVE 用）。
            - 如果为 None，则默认使用 vocab 头部 JSON 里的 line_order/order。
            - 如果显式传入一个整数，则作为手动 override，覆盖 vocab 里的设置。
        max_path_length : int
            fragment_graph 里最远考虑的最短路长度（用于 path bias）。
        fragmentizer : BaseFragmentizer | None
            如果传了一个具体的 fragmentizer 实例，则直接使用，不再根据 vocab 构造。
        frag_method : str
            碎片化方法名，目前支持 "dove" / "brics" / "brics_vanilla" / "brics_overlap"。
        overlap_degree : int | None
            == k_overlap，fragment_graph 里两个 fragment 至少重叠多少个原子才视作相连。
            - 如果为 None，则使用 vocab meta 里的 "overlap_degree"；
            - 如果 vocab 里也没有，则默认为 1（只要有一个 atom 重叠就连边）。
        """
        self.max_path_length = max_path_length
        self.frag_method = frag_method.lower()

        # 读 vocab meta：统一从 FragmentMeta 来（与 fragmentizer.py 保持一致）
        meta = FragmentMeta.from_vocab(vocab_path)

        # line_order / k_line：优先用显式传入，其次用 vocab meta
        if order is None:
            self.line_order = int(meta.line_order)
        else:
            self.line_order = int(order)

        # overlap_degree / k_overlap：优先用显式传入，其次用 vocab meta
        if overlap_degree is None:
            self.overlap_degree = int(meta.overlap_degree)
        else:
            self.overlap_degree = int(overlap_degree)

        # 记录 kekulize 设置，用于 smi_to_mol，保证和建 vocab 时一致
        self.kekulize = bool(meta.kekulize)
        self.meta = meta

        # 如果外部已经构造好了 fragmentizer，就直接用；否则通过工厂函数构造
        if fragmentizer is not None:
            self.fragmentizer = fragmentizer
        else:
            # BRICS 系列：直接走 BricsFragmentizer（不修改原来的 DOVE 工厂逻辑）
            if self.frag_method in ("brics", "brics_vanilla", "brics_overlap"):
                if self.frag_method == "brics_vanilla":
                    mode = "vanilla"
                else:
                    mode = "overlap"
                self.fragmentizer = BricsFragmentizer(
                    vocab_path=vocab_path,
                    mode=mode,
                )
            else:
                # DOVE 等原有方法，通过工厂函数构造
                self.fragmentizer = build_fragmentizer(
                    name=self.frag_method,
                    vocab_path=vocab_path,
                    order=self.line_order,
                    overlap_degree=self.overlap_degree,
                )

    def __call__(self, smiles: str) -> Dict[str, Any] | None:
        # 这里用 vocab 里的 kekulize 设置，保证和建 vocab 时一致
        mol = smi_to_mol(smiles, kekulize=self.kekulize)
        if mol is None:
            return None

        # 第三个返回值 final_details 用于分层预训练标签（DOVE 嵌套 fragment）
        group_atom_indices, fragment_vocab_ids, final_details = self.fragmentizer.tokenize(mol)

        if not group_atom_indices:
            return None

        # --------- atom-level graph ---------
        atom_graph = dgl.graph(
            data=[(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()],
            num_nodes=mol.GetNumAtoms(),
        )
        atom_graph = dgl.to_bidirected(atom_graph, copy_ndata=True)

        atom_features_list = [atom_featurizer(atom) for atom in mol.GetAtoms()]
        atom_graph.ndata["feat"] = torch.tensor(atom_features_list, dtype=torch.float32)

        if mol.GetNumBonds() > 0:
            bond_features_list = [bond_featurizer(bond) for bond in mol.GetBonds()]
            bond_feats = torch.tensor(bond_features_list, dtype=torch.float32)
            # 无向图 → 每条化学键在 DGL 里表示成两个方向的边
            atom_graph.edata["feat"] = torch.cat([bond_feats, bond_feats], dim=0)

        # --------- fragment-level graph ---------
        fragment_graph = self._build_fragment_graph(group_atom_indices)
        if fragment_vocab_ids:
            fragment_graph.ndata["id"] = torch.tensor(fragment_vocab_ids, dtype=torch.long)

        node_ids, macro_node_ids = self._separate_group_indices(group_atom_indices)

        return {
            "atom_graph": atom_graph,
            "fragment_graph": fragment_graph,
            "node_ids": node_ids,
            "macro_node_ids": macro_node_ids,
            "smiles": smiles,
            "group_atom_indices": group_atom_indices,
            "final_details": final_details,
        }

    def _separate_group_indices(
        self,
        group_idx: List[List[int] | set],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        把每个 fragment 的原子索引展开成两个一维向量：
          - node_ids       : 拼在一起的 atom 索引
          - macro_node_ids : 与 node_ids 等长，表示对应 atom 属于第几个 fragment
        """
        node_ids: List[int] = []
        macro_node_ids: List[int] = []
        for i, group in enumerate(group_idx):
            group_list = list(group)
            macro_node_ids.extend([i] * len(group_list))
            node_ids.extend(group_list)
        return torch.tensor(node_ids, dtype=torch.long), torch.tensor(macro_node_ids, dtype=torch.long)

    def _build_fragment_graph(self, group_atom_indices: List[List[int] | set]) -> dgl.DGLGraph:
        """
        构建 fragment-level 图：

          1) 先根据“原子重叠 >= k_overlap”构造一个稀疏的无向图（NetworkX）；
          2) 再在 DGL 里创建一个完全图（所有节点对都有边），
             并在 edge data 里存 shortest path 长度和实际路径（用于 path bias）。

        这里的 k_overlap = self.overlap_degree，对应 DOVE-k 里的“k-degree overlapping fragmentation”。
        """
        num_fragments = len(group_atom_indices)
        if num_fragments == 0:
            return dgl.graph(([], []), num_nodes=0)

        k_overlap = max(int(self.overlap_degree), 1)

        # 1. 先构建“物理连接图”（只放真正有重叠的边）
        adj = torch.zeros((num_fragments, num_fragments), dtype=torch.float32)
        for i in range(num_fragments):
            set_i = set(group_atom_indices[i])
            for j in range(i + 1, num_fragments):
                set_j = set(group_atom_indices[j])
                # 交集大小 >= k_overlap 才连边
                if len(set_i.intersection(set_j)) >= k_overlap:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

        nx_graph = nx.from_numpy_array(adj.numpy())

        # 2. 在 DGL 里建一个完全图（所有 fragment 对都连），方便存储 pair-wise 的 path / distance
        src, dst = torch.meshgrid(
            torch.arange(num_fragments),
            torch.arange(num_fragments),
            indexing="ij",
        )
        fragment_graph = dgl.graph((src.flatten(), dst.flatten()), num_nodes=num_fragments)

        # --- 1) shortest path LENGTHS (A^d) ---
        path_lengths = dict(nx.all_pairs_shortest_path_length(nx_graph, cutoff=self.max_path_length))
        dist_data = torch.full((num_fragments, num_fragments), -1, dtype=torch.long)
        for i, paths in path_lengths.items():
            for j, dist in paths.items():
                dist_data[i, j] = dist
        fragment_graph.edata["path"] = dist_data.flatten()

        # --- 2) full shortest PATHS (A^p) ---
        paths_dict = dict(nx.all_pairs_shortest_path(nx_graph, cutoff=self.max_path_length))
        paths_matrix = torch.full(
            (num_fragments, num_fragments, self.max_path_length + 1),
            -1,
            dtype=torch.long,
        )
        for i, paths in paths_dict.items():
            for j, path_list in paths.items():
                path_len = len(path_list)
                paths_matrix[i, j, :path_len] = torch.tensor(path_list, dtype=torch.long)

        fragment_graph.edata["paths_matrix"] = paths_matrix.reshape(-1, self.max_path_length + 1)

        return fragment_graph

