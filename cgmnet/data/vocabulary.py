# cgmnet/data/vocabulary.py
"""
Contains the FINAL, most memory-efficient implementation for building a
molecular fragment vocabulary, strictly aligned with the FragFormer DOVE algorithm.
This version uses an iterator-based parallel map to handle massive datasets.

这里主要做两件事：
  1) 使用 MolGraph + DOVE 的并行挖掘流程，统计高频 fragment。
  2) 把统计好的 fragment 写成统一格式的 vocab 文件，头部 JSON
     与 cgmnet.data.fragmentizer.FragmentMeta 完全对应：
        {
          "kekulize": bool,
          "frag_method": "dove",
          "line_order": int,       # k_line: line graph 的阶数
          "overlap_degree": int,   # k_overlap: fragment graph 的重叠阈值
          "order": int             # 兼容旧代码 = line_order
        }

扩展：
  - build_brics_vocabulary 提供了基于 BRICS 的简单词表构建（频次统计版），
    可选 vanilla / overlap 两种模式。
"""
import json
import multiprocessing as mp
from collections import defaultdict
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


from tqdm import tqdm
from rdkit import Chem

from cgmnet.data.fragmentizer import MolGraph
from cgmnet.utils.chem_utils import (
    smi_to_mol,
    cnt_atom,
    get_submol,
    mol_to_smi,
)

from cgmnet.data.frag_algos import (
    brics_partition_atom_sets,
    compute_inter_bonds_from_partition,
    apply_overlap_k1,
    rbrics_partition_atom_sets,
    recap_partition_atom_sets,
    macfrag_enumerate_atom_sets,  
    relmole_partition_atom_sets,
    relmole_apply_overlap,
    jt_partition_atom_sets,
    bemis_murcko_partition_atom_sets,
    ertl_fg_partition_atom_sets,
    louvain_partition_atom_sets, 
)


# ---------------------------------------------------------------------------
# Vocab 写入工具
# ---------------------------------------------------------------------------

def write_vocab_file(
    fragment_counts: Dict[str, int],
    output_path: str,
    kekulize: bool = False,
    frag_method: str = "dove",
    line_order: int = 1,
    overlap_degree: int | None = None,
    extra_meta: dict | None = None,
):
    """
    Write fragment_counts into a vocabulary file with a unified format.

    Header (single JSON line), at least:
        {
          "kekulize": bool,
          "frag_method": str,
          "line_order": int,
          "overlap_degree": int,
          "order": int      # backward-compatible alias for line_order
        }

    Following lines:
        "<fragment_smi>\\t<atom_num>\\t<freq>"

    Parameters
    ----------
    fragment_counts : dict[str, int]
        SMILES -> frequency.
    output_path : str
        Path to save the vocab.
    kekulize : bool
        Whether kekulization was used when generating the fragments.
    frag_method : str
        Fragmentation method name, e.g. "dove" 或 "brics_overlap" 等。
    line_order : int
        k_line: the order of line graph transforms used during mining.
    overlap_degree : int | None
        k_overlap for fragment graph; if None, defaults to line_order.
    extra_meta : dict | None
        Extra metadata to insert into the header JSON.
    """
    if overlap_degree is None:
        overlap_degree = line_order

    meta = {
        "kekulize": kekulize,
        "frag_method": frag_method,
        "line_order": int(line_order),
        "overlap_degree": int(overlap_degree),
        # 为了兼容旧版只看 "order" 的代码，这里保留一份别名
        "order": int(line_order),
    }
    if extra_meta:
        meta.update(extra_meta)

    with open(output_path, "w") as f:
        # 单行 JSON header，兼容 FragmentMeta.load_vocab_meta
        f.write(json.dumps(meta) + "\n")
        # 按 atom 数从大到小排序（和你原来 ChemBL 词表风格一致）
        for smi in sorted(fragment_counts.keys(), key=lambda s: cnt_atom(s), reverse=True):
            atom_num = cnt_atom(smi)
            freq = int(fragment_counts.get(smi, 0))
            f.write(f"{smi}\t{atom_num}\t{freq}\n")


# ---------------------------------------------------------------------------
# DOVE 挖掘的并行 worker
# ---------------------------------------------------------------------------

def _worker_get_merge_candidates(
    indexed_mol_graph: Tuple[int, MolGraph]
) -> Tuple[int, Dict[str, int], MolGraph]:
    """
    Worker function for the final DOVE implementation.

    Receives an indexed MolGraph:
        (original_index, mol_graph)
    builds adjacency, collects candidate merge fragment counts, and returns:
        (original_index, candidate_smiles_counts, updated_mol_graph)
    """
    original_index, mol_graph = indexed_mol_graph
    mol_graph.build_adj_table()
    candidate_smiles_counts = mol_graph.get_nei_smis()
    return original_index, candidate_smiles_counts, mol_graph


# ---------------------------------------------------------------------------
# 主入口：构建 DOVE vocab
# ---------------------------------------------------------------------------

def build_dove_vocabulary(
    smiles_list: List[str],
    fragments_to_mine: int,
    output_path: str,
    order: int = 1,
    n_jobs: int = 64,
    kekulize: bool = False,
):
    """
    Builds a fragment vocabulary using an iterator-based, memory-efficient
    implementation of the DOVE algorithm.

    Parameters
    ----------
    smiles_list : list[str]
        Input molecules (SMILES).
    fragments_to_mine : int
        Number of principal fragments to mine on top of the initial ones.
    output_path : str
        Where to write the vocab file.
    order : int
        == line_order (k_line): number of times to apply line graph transform
        during vocab mining. 为了兼容旧代码，参数名仍叫 order。
    n_jobs : int
        Process count for multiprocessing.Pool.
    kekulize : bool
        Whether to kekulize molecules when parsing.
    """
    line_order = int(order)

    print(f"Initializing MolGraph objects for {len(smiles_list)} molecules...")
    mol_graphs: List[MolGraph] = []
    initial_vocab_counts: Dict[str, int] = defaultdict(int)

    # Step 1: 构建 MolGraph + 统计 L^k(G) 上的初始节点 fragment 频率
    for smi in tqdm(smiles_list, desc="Processing initial molecules"):
        rdkit_mol = smi_to_mol(smi, kekulize=kekulize)
        if rdkit_mol is None:
            continue
        mg = MolGraph(rdkit_mol)
        for _ in range(line_order):
            mg.line_graph_transform()
        mg.build_init_macro_nodes()
        mol_graphs.append(mg)
        for fragment_smi, count in mg.get_nodes_smiles().items():
            if fragment_smi:
                initial_vocab_counts[fragment_smi] += count

    print(f"Found {len(initial_vocab_counts)} initial unique fragments.")

    global_vocab_counts: Dict[str, int] = initial_vocab_counts
    num_to_extract = int(fragments_to_mine)
    print(f"Target: Mining {num_to_extract} new principal fragments on top of initial ones.")

    selected_smis: List[str] = []

    # Step 2: 迭代式地从候选 merge 里选最“高频”的新 fragment
    with mp.Pool(n_jobs) as pool:
        pbar = tqdm(total=num_to_extract, desc="Mining new fragments")

        while len(selected_smis) < num_to_extract:
            print(f"\nIteration {len(selected_smis) + 1}/{num_to_extract}: Generating and aggregating candidates...")

            total_candidate_counts: Dict[str, int] = defaultdict(int)

            # 把 (index, MolGraph) 发送给各个 worker
            result_iterator = pool.imap_unordered(
                _worker_get_merge_candidates,
                enumerate(mol_graphs),
            )

            for _, (original_index, candidate_counts, updated_graph) in enumerate(
                tqdm(result_iterator, total=len(mol_graphs), desc="  - Processing molecules")
            ):
                # 更新该分子对应的 MolGraph
                mol_graphs[original_index] = updated_graph
                # 聚合候选 fragment 频率
                for smi, count in candidate_counts.items():
                    if smi:
                        total_candidate_counts[smi] += count

            if not total_candidate_counts:
                print("\nNo more merge candidates found. Stopping.")
                break

            # 候选从高频到低频排序
            sorted_candidates = sorted(
                total_candidate_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            merge_smi, max_count = None, 0
            for candidate_smi, candidate_count in sorted_candidates:
                if candidate_smi not in global_vocab_counts:
                    merge_smi, max_count = candidate_smi, candidate_count
                    break

            if merge_smi is None:
                print("\nNo more *new* fragments can be mined. Stopping.")
                break

            selected_smis.append(merge_smi)
            global_vocab_counts[merge_smi] = max_count
            pbar.update(1)

            print(f"  - Selected new fragment: {merge_smi} (Frequency: {max_count})")
            print("  - Updating all molecular graphs with the new fragment...")
            # 对所有分子图应用这个 merge_smi
            for mg in mol_graphs:
                mg.merge(merge_smi)

        pbar.close()

    print(f"\nTotal vocabulary size: {len(global_vocab_counts)}")
    print(f"Saving vocabulary to {output_path}...")

    # Step 3: 写出 vocab 文件
    # 这里 overlap_degree 暂时用 line_order 作为默认（和 FragmentMeta 里保持一致）
    write_vocab_file(
        fragment_counts=global_vocab_counts,
        output_path=output_path,
        kekulize=kekulize,
        frag_method="dove",
        line_order=line_order,
        overlap_degree=line_order,
    )

    print("Vocabulary building complete.")


# ---------------------------------------------------------------------------
# BRICS 词表构建（简单频次统计版）
# ---------------------------------------------------------------------------

def build_brics_vocabulary(
    smiles_list: List[str],
    output_path: str,
    kekulize: bool = False,
    mode: str = "overlap",
):
    """
    简单版 BRICS vocab 构建：

      - 对每个 SMILES 做 BRICS partition（可选 overlap=k1）；
      - 把每个 fragment 的 SMILES 计数；
      - 调 write_vocab_file(...) 输出统一格式。

    不做 DOVE 式的“迭代主 fragment 挖掘”，仅频次统计。
    """
    mode = mode.lower()
    if mode not in ("vanilla", "overlap"):
        raise ValueError(f"BRICS vocab mode must be 'vanilla' or 'overlap', got {mode}")

    fragment_counts: Dict[str, int] = defaultdict(int)

    for smi in tqdm(smiles_list, desc=f"Building BRICS vocab (mode={mode})"):
        mol = smi_to_mol(smi, kekulize=kekulize)
        if mol is None:
            continue

        groups_vanilla = brics_partition_atom_sets(mol)
        if not groups_vanilla:
            continue

        if mode == "overlap":
            inter_bonds = compute_inter_bonds_from_partition(mol, groups_vanilla)
            groups = apply_overlap_k1(groups_vanilla, inter_bonds)
        else:
            groups = groups_vanilla

        for atoms in groups:
            atom_list = list(atoms)
            if not atom_list:
                continue
            submol = get_submol(mol, atom_list)
            frag_smi = mol_to_smi(submol)
            if frag_smi:
                fragment_counts[frag_smi] += 1

    # BRICS 没有 line graph 概念，这里 line_order / overlap_degree 简单设为 1，
    # 主要是为了兼容 FragmentMeta / featurizer 的接口。
    extra_meta = {"mode": mode}
    # frag_method 里直接写清模式，便于后续识别
    write_vocab_file(
        fragment_counts=fragment_counts,
        output_path=output_path,
        kekulize=kekulize,
        frag_method=f"brics_{mode}",  # "brics_overlap" 或 "brics_vanilla"
        line_order=1,
        overlap_degree=1,
        extra_meta=extra_meta,
    )

    print("BRICS vocabulary building complete.")

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# r-BRICS 词表构建（简单频次统计版）
# ---------------------------------------------------------------------------

def build_rbrics_vocabulary(
    smiles_list: List[str],
    output_path: str,
    kekulize: bool = False,
    mode: str = "overlap",
):
    """
    简单版 r-BRICS vocab 构建：

      - 对每个 SMILES 做 r-BRICS partition（可选 overlap=k1）；
      - 把每个 fragment 的 SMILES 计数；
      - 调 write_vocab_file(...) 输出统一格式。

    同 BRICS 一样，这里不做 DOVE 那种“迭代主 fragment 挖掘”，只做频次统计。
    """
    mode = mode.lower()
    if mode not in ("vanilla", "overlap"):
        raise ValueError(f"r-BRICS vocab mode must be 'vanilla' or 'overlap', got {mode}")

    fragment_counts: Dict[str, int] = defaultdict(int)

    for smi in tqdm(smiles_list, desc=f"Building r-BRICS vocab (mode={mode})"):
        mol = smi_to_mol(smi, kekulize=kekulize)
        if mol is None:
            continue

        groups_vanilla = rbrics_partition_atom_sets(mol)
        if not groups_vanilla:
            continue

        if mode == "overlap":
            inter_bonds = compute_inter_bonds_from_partition(mol, groups_vanilla)
            groups = apply_overlap_k1(groups_vanilla, inter_bonds)
        else:
            groups = groups_vanilla

        for atoms in groups:
            atom_list = list(atoms)
            if not atom_list:
                continue
            submol = get_submol(mol, atom_list)
            frag_smi = mol_to_smi(submol)
            if frag_smi:
                fragment_counts[frag_smi] += 1

    extra_meta = {"mode": mode}
    # frag_method 里直接写 rbrics_{mode}，便于后续识别
    write_vocab_file(
        fragment_counts=fragment_counts,
        output_path=output_path,
        kekulize=kekulize,
        frag_method=f"rbrics_{mode}",  # "rbrics_overlap" or "rbrics_vanilla"
        line_order=1,
        overlap_degree=1,
        extra_meta=extra_meta,
    )

    print("r-BRICS vocabulary building complete.")


# ---------------------------------------------------------------------------
# RECAP 词表构建（简单频次统计版）
# ---------------------------------------------------------------------------

def build_recap_vocabulary(
    smiles_list: List[str],
    output_path: str,
    kekulize: bool = False,
    mode: str = "overlap",
):
    """
    RECAP vocab 构建（与 BRICS 类似）：

      - 对每个 SMILES 做 RECAP partition（vanilla / overlap）；
      - 统计每个 fragment SMILES 的频次；
      - 调 write_vocab_file(...) 输出统一格式。

    不做 DOVE 式迭代主 fragment 挖掘，仅频次统计。
    """
    mode = mode.lower()
    if mode not in ("vanilla", "overlap"):
        raise ValueError(f"RECAP vocab mode must be 'vanilla' or 'overlap', got {mode}")

    fragment_counts: Dict[str, int] = defaultdict(int)

    for smi in tqdm(smiles_list, desc=f"Building RECAP vocab (mode={mode})"):
        mol = smi_to_mol(smi, kekulize=kekulize)
        if mol is None:
            continue

        groups_vanilla = recap_partition_atom_sets(mol)
        if not groups_vanilla:
            continue

        if mode == "overlap":
            inter_bonds = compute_inter_bonds_from_partition(mol, groups_vanilla)
            groups = apply_overlap_k1(groups_vanilla, inter_bonds)
        else:
            groups = groups_vanilla

        for atoms in groups:
            atom_list = list(atoms)
            if not atom_list:
                continue
            submol = get_submol(mol, atom_list)
            frag_smi = mol_to_smi(submol)
            if frag_smi:
                fragment_counts[frag_smi] += 1

    extra_meta = {"mode": mode}
    # RECAP 也没有 line graph 概念，这里 line_order / overlap_degree 简单设为 1
    write_vocab_file(
        fragment_counts=fragment_counts,
        output_path=output_path,
        kekulize=kekulize,
        frag_method=f"recap_{mode}",  # "recap_overlap" 或 "recap_vanilla"
        line_order=1,
        overlap_degree=1,
        extra_meta=extra_meta,
    )

    print("RECAP vocabulary building complete.")




# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ReLMole 词表构建（简单频次统计版）
# ---------------------------------------------------------------------------

def build_relmole_vocabulary(
    smiles_list: List[str],
    output_path: str,
    kekulize: bool = False,
    mode: str = "overlap",
):
    """
    ReLMole vocab 构建（与 BRICS / RECAP 类似）：

      - 对每个 SMILES 做 ReLMole partition（vanilla / overlap）；
      - 统计每个 fragment SMILES 的频次；
      - 调 write_vocab_file(...) 输出统一格式。

    不做 DOVE 式迭代主 fragment 挖掘，仅频次统计。
    """
    mode = mode.lower()
    if mode not in ("vanilla", "overlap"):
        raise ValueError(f"ReLMole vocab mode must be 'vanilla' or 'overlap', got {mode}")

    fragment_counts: Dict[str, int] = defaultdict(int)

    for smi in tqdm(smiles_list, desc=f"Building ReLMole vocab (mode={mode})"):
        mol = smi_to_mol(smi, kekulize=kekulize)
        if mol is None:
            continue

        groups_vanilla = relmole_partition_atom_sets(mol)
        if not groups_vanilla:
            continue

        if mode == "overlap":
            groups = relmole_apply_overlap(mol, groups_vanilla)
        else:
            groups = groups_vanilla

        for atoms in groups:
            atom_list = list(atoms)
            if not atom_list:
                continue
            submol = get_submol(mol, atom_list)
            frag_smi = mol_to_smi(submol)
            if frag_smi:
                fragment_counts[frag_smi] += 1

    extra_meta = {"mode": mode}
    # ReLMole 没有 line graph 概念，这里 line_order / overlap_degree 简单设为 1
    write_vocab_file(
        fragment_counts=fragment_counts,
        output_path=output_path,
        kekulize=kekulize,
        frag_method=f"relmole_{mode}",  # "relmole_overlap" 或 "relmole_vanilla"
        line_order=1,
        overlap_degree=1,
        extra_meta=extra_meta,
    )

    print("ReLMole vocabulary building complete.")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# JT vocab 构建（clique 频次统计版）
# ---------------------------------------------------------------------------

def build_jt_vocabulary(
    smiles_list: List[str],
    output_path: str,
    kekulize: bool = False,
):
    """
    JT 风格的 vocab 构建：

      - 对每个 SMILES 做 junction-tree 分解（环 + 键的 cliques）；
      - 把每个 fragment 的 SMILES 计数；
      - 调 write_vocab_file(...) 输出统一格式。

    不做 DOVE 式“迭代主 fragment 挖掘”，仅频次统计（和 BRICS/RECAP 一样）。
    """
    fragment_counts: Dict[str, int] = defaultdict(int)

    for smi in tqdm(smiles_list, desc="Building JT vocab"):
        mol = smi_to_mol(smi, kekulize=kekulize)
        if mol is None:
            continue

        groups = jt_partition_atom_sets(mol)
        if not groups:
            continue

        for atoms in groups:
            atom_list = list(atoms)
            if not atom_list:
                continue
            submol = get_submol(mol, atom_list)
            frag_smi = mol_to_smi(submol)
            if frag_smi:
                fragment_counts[frag_smi] += 1

    # JT 没有 line graph 概念，这里 line_order / overlap_degree 简单设为 1，
    # 主要是为了兼容 FragmentMeta / CGMNetFeaturizer 接口。
    extra_meta = {"algo": "jt_vae"}
    write_vocab_file(
        fragment_counts=fragment_counts,
        output_path=output_path,
        kekulize=kekulize,
        frag_method="jt",       # vocab header 里的标记
        line_order=1,
        overlap_degree=1,
        extra_meta=extra_meta,
    )

    print("JT vocabulary building complete.")


# ---------------------------------------------------------------------------
# MacFrag 词表构建（宏碎片，简单频次统计）
# ---------------------------------------------------------------------------

def build_macfrag_vocabulary(
    smiles_list: List[str],
    output_path: str,
    kekulize: bool = False,
    frag_method: str = "macfrag",
    max_blocks: int = 4,
):
    """
    MacFrag vocab 构建：

      - 对每个 SMILES 跑 MacFrag（基于 BRICS 的 multi-scale）；
      - 只统计“宏碎片”的 SMILES 频次（base fragment 仅用来构造宏碎片）；
      - 调 write_vocab_file(...) 输出统一格式。

    不做 DOVE 式迭代挖掘，只是频次统计。
    """
    max_blocks = max(int(max_blocks), 1)
    fragment_counts: Dict[str, int] = defaultdict(int)

    for smi in tqdm(smiles_list, desc=f"Building MacFrag vocab (max_blocks={max_blocks})"):
        mol = smi_to_mol(smi, kekulize=kekulize)
        if mol is None:
            continue

        macro_items = macfrag_enumerate_atom_sets(mol, max_blocks=max_blocks)
        if not macro_items:
            continue

        for _, frag_smi in macro_items:
            if frag_smi:
                fragment_counts[frag_smi] += 1

    extra_meta = {
        "max_blocks": max_blocks,
    }

    # MacFrag 不涉及 line graph，这里 line_order/overlap_degree 设为 1，兼容 FragmentMeta
    write_vocab_file(
        fragment_counts=fragment_counts,
        output_path=output_path,
        kekulize=kekulize,
        frag_method=frag_method,  # 直接用传入的 "macfrag" / "macfrag_overlap" 等
        line_order=1,
        overlap_degree=1,
        extra_meta=extra_meta,
    )

    print("MacFrag vocabulary building complete.")



def build_bm_vocabulary(
    smiles_list: List[str],
    output_path: str,
    kekulize: bool = False,
    mode: str = "overlap",
):
    """
    Robust Bemis–Murcko vocab 构建（简单频次统计版）：

      - 对每个 SMILES 做 BM partition（vanilla 或 overlap）；
      - 统计每个 fragment SMILES 的频次；
      - 调 write_vocab_file(...) 输出统一格式。

    不做 DOVE 式的“迭代主 fragment 挖掘”，仅频次统计。
    """
    mode = mode.lower()
    if mode not in ("vanilla", "overlap"):
        raise ValueError(f"BM vocab mode must be 'vanilla' or 'overlap', got {mode}")

    fragment_counts: Dict[str, int] = defaultdict(int)

    for smi in tqdm(smiles_list, desc=f"Building BM vocab (mode={mode})"):
        mol = smi_to_mol(smi, kekulize=kekulize)
        if mol is None:
            continue

        groups_vanilla, _ = bemis_murcko_partition_atom_sets(mol)
        if not groups_vanilla:
            continue

        if mode == "overlap":
            inter_bonds = compute_inter_bonds_from_partition(mol, groups_vanilla)
            groups = apply_overlap_k1(groups_vanilla, inter_bonds)
        else:
            groups = groups_vanilla

        for atoms in groups:
            atom_list = list(atoms)
            if not atom_list:
                continue
            submol = get_submol(mol, atom_list)
            frag_smi = mol_to_smi(submol)
            if frag_smi:
                fragment_counts[frag_smi] += 1

    extra_meta = {"mode": mode}
    # BM 没有 line graph，这里 line_order / overlap_degree 简单设 1，与 BRICS/rBRICS/RECAP 对齐
    write_vocab_file(
        fragment_counts=fragment_counts,
        output_path=output_path,
        kekulize=kekulize,
        frag_method=f"bm_{mode}",   # "bm_overlap" 或 "bm_vanilla"
        line_order=1,
        overlap_degree=1,
        extra_meta=extra_meta,
    )

    print("Bemis–Murcko vocabulary building complete.")


# ---------------------------------------------------------------------------
# EFG 词表构建（简单频次统计版）
# ---------------------------------------------------------------------------

def build_efg_vocabulary(
    smiles_list: List[str],
    output_path: str,
    kekulize: bool = False,
    frag_method: str = "efgs_overlap",
):
    """
    EFG vocab 构建（与 BRICS/RECAP 类似）：

      - 对每个 SMILES 做 Ertl FG + skeleton partition；
      - 统计每个 fragment SMILES 的频次；
      - 调 write_vocab_file(...) 输出统一格式。

    不做 DOVE 式主 fragment 挖掘，仅频次统计。
    """
    fragment_counts: Dict[str, int] = defaultdict(int)

    for smi in tqdm(smiles_list, desc=f"Building EFG vocab (frag_method={frag_method})"):
        mol = smi_to_mol(smi, kekulize=kekulize)
        if mol is None:
            continue

        groups = ertl_fg_partition_atom_sets(mol)
        if not groups:
            continue

        for atoms in groups:
            atom_list = list(atoms)
            if not atom_list:
                continue
            submol = get_submol(mol, atom_list)
            frag_smi = mol_to_smi(submol)
            if frag_smi:
                fragment_counts[frag_smi] += 1

    # EFG 没有 line graph 概念，这里 line_order / overlap_degree = 1，
    # 主要是为了兼容 FragmentMeta / CGMNetFeaturizer 接口。
    write_vocab_file(
        fragment_counts=fragment_counts,
        output_path=output_path,
        kekulize=kekulize,
        frag_method=frag_method,  # 直接把 CLI 传进来的写进去，例如 "efgs_overlap"
        line_order=1,
        overlap_degree=1,
        extra_meta=None,
    )

    print("EFG vocabulary building complete.")


# ---------------------------------------------------------------------------
# Louvain 词表构建（简单频次统计版）
# ---------------------------------------------------------------------------

def build_louvain_vocabulary(
    smiles_list: List[str],
    output_path: str,
    order: int = 0,
    kekulize: bool = False,
):
    """
    Louvain vocab 构建：

      - 对每个 SMILES，用 louvain_partition_atom_sets(mol, k_line=order) 做碎片；
      - 统计每个 fragment SMILES 的频次；
      - 调 write_vocab_file(...) 输出统一格式。

    注意：
      - 这里只做“频次统计”，不做 DOVE 式迭代挖掘；
      - order == k_line == L^k(G) 的 k。
    """
    fragment_counts: Dict[str, int] = defaultdict(int)

    for smi in tqdm(smiles_list, desc=f"Building Louvain vocab (k_line={order})"):
        mol = smi_to_mol(smi, kekulize=kekulize)
        if mol is None:
            continue

        try:
            groups = louvain_partition_atom_sets(mol, k_line=order)
        except ImportError as e:
            # 直接抛出去，让脚本看到清晰的依赖错误
            raise e
        except Exception:
            # 某些极端分子失败就跳过
            continue

        if not groups:
            continue

        for atoms in groups:
            atom_list = list(atoms)
            if not atom_list:
                continue
            submol = get_submol(mol, atom_list)
            frag_smi = mol_to_smi(submol)
            if frag_smi:
                fragment_counts[frag_smi] += 1

    extra_meta = {"k_line": int(order)}

    # 对 Louvain 而言：
    #   - line_order    = k_line
    #   - overlap_degree = 1（fragment graph 里“至少 1 个原子重叠就连边”）
    write_vocab_file(
        fragment_counts=fragment_counts,
        output_path=output_path,
        kekulize=kekulize,
        frag_method="louvain",
        line_order=int(order),
        overlap_degree=1,
        extra_meta=extra_meta,
    )

    print("Louvain vocabulary building complete.")


# ---------------------------------------------------------------------------
# 统一入口：根据 frag_method 分发到不同的 vocab 构建算法
# ---------------------------------------------------------------------------

def build_vocabulary(
    smiles_list: List[str],
    fragments_to_mine: int,
    output_path: str,
    frag_method: str = "dove",
    order: int = 1,
    n_jobs: int = 64,
    kekulize: bool = False,
):
    """
    Unified vocabulary builder.

    当前实现：
      - "dove"           : DOVE 算法（迭代挖主 fragment）
      - "brics"          : BRICS + overlap=k1（默认）
      - "brics_vanilla"  : BRICS 严格 partition（无重叠）
      - "brics_overlap"  : 显式 BRICS + overlap=k1

    Parameters
    ----------
    smiles_list : list[str]
        输入 SMILES 列表。
    fragments_to_mine : int
        要在初始 fragment 基础上额外挖掘多少个“主” fragment（仅 DOVE 使用）。
    output_path : str
        词表输出路径。
    frag_method : str
        碎片化方法名。
    order : int
        对于 DOVE：k_line，line graph transform 的阶数。
    n_jobs : int
        并行进程数（仅 DOVE 使用）。
    kekulize : bool
        词表挖掘时是否 kekulize。
    """
    frag_method = frag_method.lower()

    if frag_method == "dove":
        return build_dove_vocabulary(
            smiles_list=smiles_list,
            fragments_to_mine=fragments_to_mine,
            output_path=output_path,
            order=order,
            n_jobs=n_jobs,
            kekulize=kekulize,
        )
    elif frag_method in ("brics", "brics_vanilla", "brics_overlap"):
        # brics / brics_overlap 默认 overlap 模式；brics_vanilla 则使用严格 partition
        if frag_method == "brics_vanilla":
            mode = "vanilla"
        else:
            mode = "overlap"

        return build_brics_vocabulary(
            smiles_list=smiles_list,
            output_path=output_path,
            kekulize=kekulize,
            mode=mode,
        )


    elif frag_method in ("rbrics", "rbrics_vanilla", "rbrics_overlap"):
        # r-BRICS 分支：rbrics / rbrics_overlap 默认 overlap 模式；rbrics_vanilla 使用严格 partition
        if frag_method == "rbrics_vanilla":
            mode = "vanilla"
        else:
            mode = "overlap"

        return build_rbrics_vocabulary(
            smiles_list=smiles_list,
            output_path=output_path,
            kekulize=kekulize,
            mode=mode,
        )


    elif frag_method in ("recap", "recap_vanilla", "recap_overlap"):
        # recap / recap_overlap 默认 overlap；recap_vanilla 使用严格 partition
        if frag_method == "recap_vanilla":
            mode = "vanilla"
        else:
            mode = "overlap"

        return build_recap_vocabulary(
            smiles_list=smiles_list,
            output_path=output_path,
            kekulize=kekulize,
            mode=mode,
        )

    elif frag_method in ("relmole", "relmole_vanilla", "relmole_overlap"):
        # relmole / relmole_overlap 默认 overlap；relmole_vanilla 使用严格 partition
        if frag_method == "relmole_vanilla":
            mode = "vanilla"
        else:
            mode = "overlap"

        return build_relmole_vocabulary(
            smiles_list=smiles_list,
            output_path=output_path,
            kekulize=kekulize,
            mode=mode,
        )



    elif frag_method in ("jt", "jt_vae", "jtvae", "junction_tree"):
        return build_jt_vocabulary(
            smiles_list=smiles_list,
            output_path=output_path,
            kekulize=kekulize,
        )


    elif frag_method in ("macfrag", "macfrag_vanilla", "macfrag_overlap"):
        # 这里约定：order == max_blocks（宏碎片最多包含几个 BRICS 基块）
        max_blocks = max(int(order), 1)

        return build_macfrag_vocabulary(
            smiles_list=smiles_list,
            output_path=output_path,
            kekulize=kekulize,
            frag_method=frag_method,   # 头部 JSON 里的 "frag_method"
            max_blocks=max_blocks,
        )


    elif frag_method in ("bm", "bm_vanilla", "bm_overlap"):
        # bm / bm_overlap 默认 overlap 模式；bm_vanilla 则使用严格 partition
        if frag_method == "bm_vanilla":
            mode = "vanilla"
        else:
            mode = "overlap"

        return build_bm_vocabulary(
            smiles_list=smiles_list,
            output_path=output_path,
            kekulize=kekulize,
            mode=mode,
        )


    elif frag_method in (
        "efg", "efg_vanilla", "efg_overlap",
        "efgs", "efgs_vanilla", "efgs_overlap",
    ):
        # 约定：XXX_vanilla → vanilla，其余都走 overlap
        if frag_method.endswith("_vanilla"):
            mode = "vanilla"
        else:
            mode = "overlap"

        # 用原始 frag_method 进入 header，方便追踪
        fm_for_header = frag_method

        return build_efg_vocabulary(
            smiles_list=smiles_list,
            output_path=output_path,
            kekulize=kekulize,
            frag_method=fm_for_header,
        )


    elif frag_method == "louvain":
        return build_louvain_vocabulary(
            smiles_list=smiles_list,
            output_path=output_path,
            order=order,
            kekulize=kekulize,
        )



    raise NotImplementedError(
        f"Fragmentation method '{frag_method}' is not implemented in build_vocabulary yet."
    )

