# cgmnet/data/atom_featurizer.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional

import torch
import dgl
import networkx as nx
from rdkit import Chem

from cgmnet.utils.chem_utils import smi_to_mol, atom_featurizer, bond_featurizer


class CGMNetAtomFeaturizer:
    """
    Atom-level variant:
      - fragment_graph 的节点 == atom
      - fragment_graph 的“物理连接图”来自 atom bond adjacency
      - 然后转成 complete graph，并在 edata 里写 path / paths_matrix（与你现有一致）
      - fragment token id 直接用原子序数 atomic_num（范围 1..118），UNK 不需要
    """

    def __init__(
        self,
        max_path_length: int = 5,
        kekulize: bool = False,
        atom_id_offset: int = 0,
    ):
        self.max_path_length = int(max_path_length)
        self.kekulize = bool(kekulize)
        # 让 token id 可以从 0 开始（比如 atomic_num-1）
        self.atom_id_offset = int(atom_id_offset)

    def __call__(self, smiles: str) -> Dict[str, Any] | None:
        mol = smi_to_mol(smiles, kekulize=self.kekulize)
        if mol is None:
            return None

        n = mol.GetNumAtoms()
        if n == 0:
            return None

        # --------- atom-level graph ---------
        atom_graph = dgl.graph(
            data=[(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()],
            num_nodes=n,
        )
        atom_graph = dgl.to_bidirected(atom_graph, copy_ndata=True)

        atom_features_list = [atom_featurizer(atom) for atom in mol.GetAtoms()]
        atom_graph.ndata["feat"] = torch.tensor(atom_features_list, dtype=torch.float32)

        if mol.GetNumBonds() > 0:
            bond_features_list = [bond_featurizer(bond) for bond in mol.GetBonds()]
            bond_feats = torch.tensor(bond_features_list, dtype=torch.float32)
            atom_graph.edata["feat"] = torch.cat([bond_feats, bond_feats], dim=0)

        # --------- fragment-level graph (atom-as-fragment) ---------
        fragment_graph = self._build_fragment_graph_from_atom_graph(mol)

        # token id：原子序数
        atom_ids = []
        for atom in mol.GetAtoms():
            aid = int(atom.GetAtomicNum()) + self.atom_id_offset
            atom_ids.append(aid)
        fragment_graph.ndata["id"] = torch.tensor(atom_ids, dtype=torch.long)

        # group_atom_indices：每个 fragment 只有一个 atom
        group_atom_indices = [{i} for i in range(n)]
        node_ids = torch.arange(n, dtype=torch.long)        # 0..n-1
        macro_node_ids = torch.arange(n, dtype=torch.long)  # 对应 fragment index

        # final_details：无层级，用 [id] 占位即可兼容你 collator
        final_details = [[int(a)] for a in atom_ids]

        return {
            "atom_graph": atom_graph,
            "fragment_graph": fragment_graph,
            "node_ids": node_ids,
            "macro_node_ids": macro_node_ids,
            "smiles": smiles,
            "group_atom_indices": group_atom_indices,
            "final_details": final_details,
        }

    def _build_fragment_graph_from_atom_graph(self, mol: Chem.Mol) -> dgl.DGLGraph:
        n = mol.GetNumAtoms()
        if n == 0:
            return dgl.graph(([], []), num_nodes=0)

        # 1) “物理连接图” = atom bond adjacency
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(range(n))
        for bond in mol.GetBonds():
            nx_graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

        # 2) complete graph + shortest path info（与你现有 featurizer 一致）
        src, dst = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        g = dgl.graph((src.flatten(), dst.flatten()), num_nodes=n)

        path_lengths = dict(nx.all_pairs_shortest_path_length(nx_graph, cutoff=self.max_path_length))
        dist_data = torch.full((n, n), -1, dtype=torch.long)
        for i, paths in path_lengths.items():
            for j, dist in paths.items():
                dist_data[i, j] = dist
        g.edata["path"] = dist_data.flatten()

        paths_dict = dict(nx.all_pairs_shortest_path(nx_graph, cutoff=self.max_path_length))
        paths_matrix = torch.full((n, n, self.max_path_length + 1), -1, dtype=torch.long)
        for i, paths in paths_dict.items():
            for j, path_list in paths.items():
                L = len(path_list)
                paths_matrix[i, j, :L] = torch.tensor(path_list, dtype=torch.long)
        g.edata["paths_matrix"] = paths_matrix.reshape(-1, self.max_path_length + 1)

        return g

