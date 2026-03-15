# cgmnet/data/fragmentizer.py
"""
Defines the abstract base class for molecular fragmentation and a concrete
implementation based on the DOVE methodology (line graph transformation and
principal subgraph mining).

本文件现在做了两件解耦：
  1) 明确区分：
       - line_order      : k-order line graph 的 k
       - overlap_degree  : k-degree overlapping fragmentation 的 k
  2) 所有与碎片相关的超参数都从 vocab 文件第一行 JSON 里读，
     通过 load_vocab_meta / FragmentMeta 统一管理。
"""
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Set, Tuple, Dict, Optional

import networkx as nx
from rdkit import Chem

from cgmnet.utils.chem_utils import get_submol, mol_to_smi


# ---------------------------------------------------------------------------
# Vocab meta / config 相关工具
# ---------------------------------------------------------------------------

@dataclass
class FragmentMeta:
    """Meta information parsed from the first line of a vocab file."""
    frag_method: str = "dove"
    line_order: int = 1          # k_line: how many times to apply line-graph transform
    overlap_degree: int = 1      # k_overlap: |Vi ∩ Vj| >= k_overlap to be neighbors
    kekulize: bool = False

    @classmethod
    def from_vocab(cls, vocab_path: str) -> "FragmentMeta":
        return load_vocab_meta(vocab_path)


def load_vocab_meta(vocab_path: str) -> FragmentMeta:
    """
    从 vocab 文件第一行 JSON 解析出 FragmentMeta。
    兼容旧格式：
        {"kekulize": false, "frag_method": "dove", "order": 1}
    新推荐格式：
        {"kekulize": false, "frag_method": "dove",
         "line_order": 1, "overlap_degree": 1}
    """
    with open(vocab_path, "r") as f:
        first_line = f.readline().strip()

    try:
        cfg = json.loads(first_line) if first_line else {}
    except json.JSONDecodeError:
        cfg = {}

    frag_method = cfg.get("frag_method", "dove")

    # 兼容：老 vocab 用 "order" 表示 line_order
    line_order = cfg.get("line_order", cfg.get("order", 1))

    # 默认 overlap_degree = line_order（对应定理里的 k）
    overlap_degree = cfg.get("overlap_degree", line_order)

    kekulize = bool(cfg.get("kekulize", False))

    return FragmentMeta(
        frag_method=frag_method,
        line_order=int(line_order),
        overlap_degree=int(overlap_degree),
        kekulize=kekulize,
    )


# ---------------------------------------------------------------------------
# 分子图工具（和 DOVE 算法有关）
# ---------------------------------------------------------------------------

class MolGraph:
    """
    A helper class to represent a molecule as a graph and perform various
    graph transformations required for DOVE vocabulary mining.
    """
    def __init__(self, mol: Chem.Mol):
        self.mol = mol
        self.nodes: List[Set[int]] = [{i} for i in range(mol.GetNumAtoms())]
        self.edges: List[Tuple[int, int]] = [
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
        ]
        self.adj_table = defaultdict(list)
        self.macro_nodes_dict: Dict[int, Tuple[List[int], Set[int]]] = {}
        self.nodeID_to_macroID: Dict[int, int] = {}
        self.smi2macro_ids = defaultdict(list)
        self.macro_nodes_id = 0
        self.distance: List[List[int]] = []
        self.paths: List[List[int]] = []

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    def update_graph(self, nodes: List[Set[int]], edges: List[Tuple[int, int]]):
        self.nodes = nodes
        self.edges = edges

    def build_adj_table(self):
        """Builds an adjacency list representation of the graph."""
        self.adj_table = defaultdict(list)
        for u, v in self.edges:
            self.adj_table[u].append(v)
            self.adj_table[v].append(u)

    def build_init_macro_nodes(self):
        """Initializes macro-nodes, where each node is a fragment."""
        self.macro_nodes_dict = {i: ([i], self.nodes[i]) for i in range(self.num_nodes)}
        self.nodeID_to_macroID = {i: i for i in range(self.num_nodes)}
        self.macro_nodes_id = self.num_nodes

    def get_nodes_smiles(self) -> Dict[str, int]:
        """Gets the SMILES of all current nodes (fragments) and their counts."""
        cnt_smi = defaultdict(int)
        for node_atoms in self.nodes:
            try:
                submol = get_submol(self.mol, list(node_atoms))
                smi = mol_to_smi(submol)
                cnt_smi[smi] += 1
            except Exception:
                continue
        return cnt_smi

    def merge_macro_nodes(self, macro_node_id1: int, macro_node_id2: int) -> str:
        """Merges two macro-nodes and returns the SMILES of the resulting fragment."""
        atom_idxs = self.macro_nodes_dict[macro_node_id1][1] | self.macro_nodes_dict[macro_node_id2][1]
        submol = get_submol(self.mol, list(atom_idxs))
        return mol_to_smi(submol)

    def get_nei_smis(self) -> Dict[str, int]:
        """
        For DOVE vocabulary mining:
        Finds all potential new fragments by merging adjacent macro-nodes.
        """
        cnt_smi = defaultdict(int)
        cnt_set = set()
        self.smi2macro_ids = defaultdict(list)

        for macro_node_id in self.macro_nodes_dict:
            node_ids, _ = self.macro_nodes_dict[macro_node_id]
            nei_macro_nodes = set()
            for node_id in node_ids:
                for nei_id in self.adj_table.get(node_id, []):
                    if nei_id not in node_ids:
                        nei_macro_nodes.add(self.nodeID_to_macroID[nei_id])

            for nei_macro_node_id in nei_macro_nodes:
                key = tuple(sorted((macro_node_id, nei_macro_node_id)))
                if key in cnt_set:
                    continue
                cnt_set.add(key)
                try:
                    smi = self.merge_macro_nodes(macro_node_id, nei_macro_node_id)
                    if smi:  # Ensure the generated SMILES is not empty
                        cnt_smi[smi] += 1
                        self.smi2macro_ids[smi].append(key)
                except Exception:
                    # If RDKit fails to process the merged structure, just skip it.
                    continue
        return cnt_smi

    def merge(self, smi: str):
        """Performs a merge operation for a given SMILES during vocab mining."""
        if smi not in self.smi2macro_ids:
            return

        for macro_node_id1, macro_node_id2 in self.smi2macro_ids[smi]:
            if macro_node_id1 not in self.macro_nodes_dict or macro_node_id2 not in self.macro_nodes_dict:
                continue

            atom_idxs = self.macro_nodes_dict[macro_node_id1][1] | self.macro_nodes_dict[macro_node_id2][1]
            node_idxs = self.macro_nodes_dict[macro_node_id1][0] + self.macro_nodes_dict[macro_node_id2][0]

            self.macro_nodes_dict.pop(macro_node_id1)
            self.macro_nodes_dict.pop(macro_node_id2)

            new_macro_node_id = self.macro_nodes_id
            self.macro_nodes_dict[new_macro_node_id] = (node_idxs, atom_idxs)

            for node_id in node_idxs:
                self.nodeID_to_macroID[node_id] = new_macro_node_id
            self.macro_nodes_id += 1

    def line_graph_transform(self):
        """
        Single-step line graph transform G -> L(G).
        """
        if not self.edges:
            return

        edge_to_new_node_id = {}
        new_nodes: List[Set[int]] = []
        for u, v in self.edges:
            edge_key = tuple(sorted((u, v)))
            if edge_key not in edge_to_new_node_id:
                edge_to_new_node_id[edge_key] = len(new_nodes)
                # new node corresponds to the union of the two endpoint atom sets
                new_nodes.append(self.nodes[u] | self.nodes[v])

        atom_to_new_nodes = defaultdict(list)
        for u, v in self.edges:
            edge_key = tuple(sorted((u, v)))
            new_node_id = edge_to_new_node_id[edge_key]
            atom_to_new_nodes[u].append(new_node_id)
            atom_to_new_nodes[v].append(new_node_id)

        new_edges = set()
        for incident_new_nodes in atom_to_new_nodes.values():
            for i in range(len(incident_new_nodes)):
                for j in range(i + 1, len(incident_new_nodes)):
                    u_new, v_new = incident_new_nodes[i], incident_new_nodes[j]
                    new_edges.add(tuple(sorted((u_new, v_new))))

        self.update_graph(new_nodes, list(new_edges))

    def build_fragment_graph(self, fragment_atom_indices: List[Set[int]], overlap_degree: int = 1):
        """
        Builds a fragment-level graph on the original molecular graph:
          - Nodes: fragments V_i
          - Edges: A_ij = 1 if |V_i ∩ V_j| >= overlap_degree
        默认 overlap_degree=1，等价于论文里的 k-degree overlapping fragmentation (k=1)。
        """
        num_fragments = len(fragment_atom_indices)
        new_edges: List[Tuple[int, int]] = []
        for i in range(num_fragments):
            new_edges.append((i, i))  # self-loop
            for j in range(i + 1, num_fragments):
                if len(fragment_atom_indices[i].intersection(fragment_atom_indices[j])) >= overlap_degree:
                    new_edges.extend([(i, j), (j, i)])
        self.edges = new_edges

    def to_complete_graph(self, max_path_length: int = 5):
        """
        Computes all-pairs shortest paths and converts the graph to a complete graph
        where edge attributes store path information.
        """
        if not self.nodes:
            return

        nx_graph = nx.Graph(self.edges)
        nx_graph.add_nodes_from(range(self.num_nodes))

        paths_dict = dict(nx.all_pairs_shortest_path(nx_graph, cutoff=max_path_length))

        new_edges, distances, paths = [], [], []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                new_edges.append((i, j))
                if i in paths_dict and j in paths_dict[i]:
                    path = paths_dict[i][j]
                    distances.append(len(path) - 1)
                    padded_path = path + [-1] * (max_path_length + 1 - len(path))
                    paths.append(padded_path)
                else:
                    distances.append(-1)
                    paths.append([-1] * (max_path_length + 1))

        self.edges = new_edges
        self.distance = distances
        self.paths = paths


# ---------------------------------------------------------------------------
# 抽象基类 & DOVE 实现
# ---------------------------------------------------------------------------

class BaseFragmentizer(ABC):
    """Abstract base class for all molecular fragmentizers."""
    @abstractmethod
    def tokenize(self, mol: Chem.Mol) -> Tuple[List[Set[int]], List[int], List[List[int]]]:
        """
        Tokenizes a molecule into a set of fragments.

        Returns
        -------
        group_atom_indices : list[set[int]]
            每个 fragment 在原子图 G 上包含哪些 atom indices。
        fragment_vocab_ids : list[int]
            每个 fragment 的“主” vocab id（用于 MCP 任务）。
        final_details : list[list[int]]
            每个 fragment 的层级 vocab id 列表（用于 nested pretraining）。
        """
        raise NotImplementedError


class DoveFragmentizer(BaseFragmentizer):
    """
    A fragmentizer implementing the DOVE algorithm. It uses a pre-computed
    vocabulary to greedily merge subgraphs based on fragment frequency.

    Args
    ----
    vocab_path : str
        Path to the vocabulary file.
    line_order / order : int, optional
        k_line: how many times to apply line-graph transform.
        `order` 参数保留是为了兼容旧代码，会被映射到 line_order。
    overlap_degree : int, optional
        k_overlap: minimum shared-atom count for two fragments to be neighbors
        when constructing the fragment graph on the original molecule.
        对 DOVE 而言，通常设置为 line_order。
    """
    def __init__(
        self,
        vocab_path: str,
        line_order: Optional[int] = None,
        overlap_degree: Optional[int] = None,
        order: Optional[int] = None,  # backward-compat
    ):
        meta = load_vocab_meta(vocab_path)

        # 兼容：优先使用显式传入的 line_order / order，其次用 vocab 里的
        if line_order is None:
            line_order = order if order is not None else meta.line_order
        if overlap_degree is None:
            overlap_degree = meta.overlap_degree

        self.vocab_path = vocab_path
        self.line_order = int(line_order)
        self.overlap_degree = int(overlap_degree)
        self.kekulize = meta.kekulize
        self.meta = meta

        # 读 vocab 列表
        with open(vocab_path, 'r') as f:
            lines = f.read().strip().split('\n')

        self.vocab_freq: Dict[str, int] = {}
        self.smi_to_id: Dict[str, int] = {}
        self.id_to_smi: List[str] = []

        # 第一行已经被 meta 消化过，这里从第二行开始
        for line in lines[1:]:
            if not line.strip():
                continue
            smi, _, freq = line.strip().split('\t')
            if smi not in self.smi_to_id:
                self.smi_to_id[smi] = len(self.id_to_smi)
                self.id_to_smi.append(smi)
            self.vocab_freq[smi] = int(freq)

        # UNK token
        self.unk_token = '<unk>'
        if self.unk_token not in self.smi_to_id:
            self.smi_to_id[self.unk_token] = len(self.id_to_smi)
            self.id_to_smi.append(self.unk_token)
        self.unk_id = self.smi_to_id[self.unk_token]

    def get_smi_id(self, smi: str) -> int:
        return self.smi_to_id.get(smi, self.unk_id)

    def tokenize(self, mol: Chem.Mol) -> Tuple[List[Set[int]], List[int], List[List[int]]]:
        """
        在原子图 G 上执行：
          1) 做 line_order 次 line graph 变换得到 L^k(G)；
          2) 在 L^k(G) 上做基于 vocab 频率的贪心合并，得到一组不重叠子图；
          3) 把每个子图映射回原子集合，作为 G 上的 fragment；
          4) 对每个 fragment 生成主 token id 和 nested id 列表。
        """
        mol_graph = MolGraph(mol)
        for _ in range(self.line_order):
            mol_graph.line_graph_transform()

        # 初始：每个节点单独一个 macro node
        macro_nodes: Dict[int, Tuple[List[int], Set[int]]] = {
            i: ([i], node_atoms) for i, node_atoms in enumerate(mol_graph.nodes)
        }
        node_to_macro_id = {i: i for i in range(mol_graph.num_nodes)}

        # final_details 用来记录“这个 fragment 包含了哪些 vocab id”（从小到大 merge 的过程）
        fragment_details: Dict[int, List[int]] = defaultdict(list)
        for i, (_, node_atoms) in macro_nodes.items():
            smi = mol_to_smi(get_submol(mol, list(node_atoms)))
            fragment_details[i].append(self.get_smi_id(smi))

        # 迭代：在 L^k(G) 上贪心合并 macro nodes
        while True:
            adj_table = defaultdict(list)
            for u, v in mol_graph.edges:
                adj_table[u].append(v)
                adj_table[v].append(u)

            potential_merges: Dict[Tuple[int, int], Tuple[str, int]] = {}
            for macro_id1, (nodes1, atoms1) in macro_nodes.items():
                neighbors = {
                    node_to_macro_id[neighbor_node_id]
                    for node_id in nodes1
                    for neighbor_node_id in adj_table.get(node_id, [])
                }

                for macro_id2 in neighbors:
                    if macro_id1 >= macro_id2:
                        continue
                    atoms2 = macro_nodes[macro_id2][1]
                    smi = mol_to_smi(get_submol(mol, list(atoms1 | atoms2)))
                    if smi in self.vocab_freq:
                        potential_merges[(macro_id1, macro_id2)] = (smi, self.vocab_freq[smi])

            if not potential_merges:
                break

            (best_id1, best_id2), (merged_smi, _) = max(
                potential_merges.items(),
                key=lambda item: item[1][1]
            )

            nodes1, atoms1 = macro_nodes.pop(best_id1)
            nodes2, atoms2 = macro_nodes.pop(best_id2)

            new_id = (max(macro_nodes.keys()) + 1) if macro_nodes else 0
            new_nodes = nodes1 + nodes2
            new_atoms = atoms1 | atoms2
            macro_nodes[new_id] = (new_nodes, new_atoms)

            for node_id in new_nodes:
                node_to_macro_id[node_id] = new_id

            merged_smi_id = self.get_smi_id(merged_smi)
            fragment_details[new_id] = (
                fragment_details.pop(best_id1)
                + fragment_details.pop(best_id2)
                + [merged_smi_id]
            )

        # 根据节点编号顺序排序 fragment，保证稳定性
        sorted_macro_nodes = sorted(macro_nodes.items(), key=lambda item: min(item[1][0]))

        group_atom_indices: List[Set[int]] = [atoms for _, (_, atoms) in sorted_macro_nodes]
        fragment_vocab_ids: List[int] = [
            self.get_smi_id(mol_to_smi(get_submol(mol, list(atoms))))
            for atoms in group_atom_indices
        ]
        final_details: List[List[int]] = [
            fragment_details[macro_id] for macro_id, _ in sorted_macro_nodes
        ]

        return group_atom_indices, fragment_vocab_ids, final_details


# ---------------------------------------------------------------------------
# 工厂函数：将来可以在这里接 BRICS / RECAP 等
# ---------------------------------------------------------------------------

def build_fragmentizer(
    name: str,
    vocab_path: str,
    order: Optional[int] = None,
    overlap_degree: Optional[int] = None,
    **kwargs,
) -> BaseFragmentizer:
    """
    Factory function to build a fragmentizer instance.

    Parameters
    ----------
    name : str
        Fragmentation method name, such as 'dove', and in the future
        'brics', 'recap', 'macfrag', etc.
    vocab_path : str
        Path to the fragment vocabulary file (for vocab-based methods like DOVE).
    order : int | None
        Backward-compatible alias for line_order of DOVE. If None, will use
        line_order from vocab meta.
    overlap_degree : int, optional
        k_overlap; if None, will fall back to vocab meta's setting.

    Notes
    -----
    目前只实现了 DOVE。其它方法接入时，只要实现 BaseFragmentizer 接口即可。
    """
    name = name.lower()
    if name == "dove":
        # 兼容：这里仍然把 order 传进去，由 DoveFragmentizer 映射到 line_order
        return DoveFragmentizer(
            vocab_path=vocab_path,
            order=order,
            overlap_degree=overlap_degree,
        )
    elif name in ("brics", "brics_vanilla", "brics_overlap"):
        # 懒加载，避免循环 import
        from cgmnet.data.frag_algos import BricsFragmentizer

        # 用方法名决定 BRICS 的 mode：默认 overlap
        if name == "brics_vanilla":
            mode = "vanilla"
        else:
            mode = "overlap"

        return BricsFragmentizer(
            vocab_path=vocab_path,
            mode=mode,
        )


# -----------------------------------------------
    elif name in ("rbrics", "rbrics_vanilla", "rbrics_overlap"):
        # 懒加载，避免循环 import
        from cgmnet.data.frag_algos import RBricsFragmentizer

        if name == "rbrics_vanilla":
            mode = "vanilla"
        else:
            mode = "overlap"

        return RBricsFragmentizer(
            vocab_path=vocab_path,
            mode=mode,
        )

# ----------------------------------------------
    elif name in ("recap", "recap_vanilla", "recap_overlap"):
        # 懒加载，避免循环 import
        from cgmnet.data.frag_algos import RecapFragmentizer

        if name == "recap_vanilla":
            mode = "vanilla"
        else:
            mode = "overlap"

        return RecapFragmentizer(
            vocab_path=vocab_path,
            mode=mode,
        )


# ----------------------------------------------
    elif name in ("relmole", "relmole_vanilla", "relmole_overlap"):
        # 懒加载，避免循环 import
        from cgmnet.data.frag_algos import ReLMoleFragmentizer

        if name == "relmole_vanilla":
            mode = "vanilla"
        else:
            mode = "overlap"

        return ReLMoleFragmentizer(
            vocab_path=vocab_path,
            mode=mode,
        )

# ---------------------------------------------



# --------------------------------------------
    elif name in ("jt", "jt_vae", "jtvae", "junction_tree"):
        from cgmnet.data.frag_algos import JTFragmentizer
        return JTFragmentizer(vocab_path=vocab_path)

# -------------------------------------------
    elif name in ("macfrag", "macfrag_vanilla", "macfrag_overlap"):
        # 懒加载，避免循环 import
        from cgmnet.data.frag_algos import MacFragFragmentizer

        # 把 order 解释为 max_blocks：如果 vocab 头部 JSON 里写了 line_order，
        # CGMNetFeaturizer 会先读 meta.line_order，再被 CLI --order 覆盖。
        max_blocks = order if order is not None and order > 0 else 4

        return MacFragFragmentizer(
            vocab_path=vocab_path,
            max_blocks=max_blocks,
        )

# ----------------------------------------------
    elif name in ("bm", "bm_vanilla", "bm_overlap"):
        from cgmnet.data.frag_algos import BmFragmentizer

        if name == "bm_vanilla":
            mode = "vanilla"
        else:
            mode = "overlap"

        return BmFragmentizer(
            vocab_path=vocab_path,
            mode=mode,
        )

# --------------------------------------------------------
    elif name in (
        "efg", "efg_vanilla", "efg_overlap",
        "efgs", "efgs_vanilla", "efgs_overlap",
    ):
        # 懒加载，避免循环 import
        from cgmnet.data.frag_algos import EfgFragmentizer

        # 约定：*_vanilla → vanilla，其余 → overlap
        if name.endswith("_vanilla"):
            mode = "vanilla"
        else:
            mode = "overlap"

        return EfgFragmentizer(
            vocab_path=vocab_path,
            mode=mode,
        )

# --------------------------------------------------------
    elif name == "louvain":
        # 懒加载避免循环 import
        from cgmnet.data.frag_algos import LouvainFragmentizer

        # 这里的 order 对 DOVE 是 k_line，
        # 对 Louvain 我们也把它当作 L^k(G) 的 k_line 用
        k_line = 0 if order is None else int(order)

        return LouvainFragmentizer(
            vocab_path=vocab_path,
            k_line=k_line,
        )


    # TODO: extend here for other methods (BRICS/RECAP/MacFrag, etc.)
    # elif name == "recap":
    #     return RecapFragmentizer(vocab_path=vocab_path, **kwargs)

    raise ValueError(
        f"Unknown fragmentizer '{name}'. "
        f"Currently supported: ['dove', 'brics', 'brics_vanilla', 'brics_overlap']. "
        f"Extend build_fragmentizer in cgmnet.data.fragmentizer to add a new method."
    )

