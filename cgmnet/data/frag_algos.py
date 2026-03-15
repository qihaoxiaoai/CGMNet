# cgmnet/data/frag_algos.py
"""
Concrete fragmentation algorithms for CGMNet.

- DOVE 仍然保留在 cgmnet.data.fragmentizer 中，不动原始实现。
- 本文件放其他碎片化算法的具体实现，例如：
    - BricsFragmentizer (vanilla / overlap)
    - 以后可以加入 EFG, AccFG 等

所有类都遵守 BaseFragmentizer 接口：
    tokenize(mol: Chem.Mol) -> (group_atom_indices, fragment_vocab_ids, final_details)
"""
from __future__ import annotations

import re
import copy
from collections import defaultdict
import networkx as nx 

from typing import List, Set, Tuple, Dict

from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold 

try:
    import community as community_louvain  # python-louvain
except ImportError:
    community_louvain = None

try:
    import efgs
except ImportError:
    efgs = None

from cgmnet.data.fragmentizer import BaseFragmentizer  # 只拿接口，不修改原实现
from cgmnet.utils.chem_utils import get_submol, mol_to_smi



# --------------------------------------------------------------------------
# extension
# --------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# BRICS helpers: vanilla partition + overlap=k1
# ---------------------------------------------------------------------------

def brics_partition_atom_sets(mol: Chem.Mol) -> List[Set[int]]:
    """
    对无显式氢的 RDKit Mol 运行 BRICS partition，返回每个 fragment 的原子 index 集合。

    Parameters
    ----------
    mol : Chem.Mol
        RDKit 分子，假定已经 RemoveHs，atom index 与 featurizer 使用的一致。

    Returns
    -------
    groups : list[set[int]]
        每个元素是一组原子索引（对应原始 mol 的 atom index）。
    """
    # 1. 给原分子上的每个原子打上 orig_idx 标签
    for atom in mol.GetAtoms():
        atom.SetIntProp("orig_idx", atom.GetIdx())

    # 2. BRICS 断键（产生 Dummy atom）
    fragmented = BRICS.BreakBRICSBonds(mol)

    # 3. 获取切块索引（在 fragmented 上的 atom idx）
    frag_tuples = Chem.GetMolFrags(fragmented, asMols=False)

    groups: List[Set[int]] = []
    for frag_indices in frag_tuples:
        cur: Set[int] = set()
        for idx in frag_indices:
            atom = fragmented.GetAtomWithIdx(idx)
            # Dummy 原子没有 orig_idx，跳过
            if atom.HasProp("orig_idx"):
                cur.add(atom.GetIntProp("orig_idx"))
        if cur:
            groups.append(cur)

    return groups


def compute_inter_bonds_from_partition(
    mol: Chem.Mol,
    groups_vanilla: List[Set[int]],
) -> List[Tuple[int, int, int, int]]:
    """
    基于 vanilla BRICS partition，计算跨 fragment 键。

    Parameters
    ----------
    mol : Chem.Mol
        原始 RDKit 分子（无显式 H）。
    groups_vanilla : list[set[int]]
        vanilla BRICS partition 的 fragment 原子集合。

    Returns
    -------
    inter_bonds : list[tuple[int,int,int,int]]
        列表元素为 (a, b, fa, fb)：
          a, b  : bond 两端原子 index
          fa, fb: 对应 fragment id（在 groups_vanilla 中的下标）
    """
    atom_to_frag: Dict[int, int] = {}
    for fid, atoms in enumerate(groups_vanilla):
        for a in atoms:
            atom_to_frag[a] = fid

    inter_bonds: List[Tuple[int, int, int, int]] = []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        fa = atom_to_frag.get(a)
        fb = atom_to_frag.get(b)
        if fa is not None and fb is not None and fa != fb:
            inter_bonds.append((a, b, fa, fb))
    return inter_bonds


def apply_overlap_k1(
    groups_vanilla: List[Set[int]],
    inter_bonds: List[Tuple[int, int, int, int]],
) -> List[Set[int]]:
    """
    基于 vanilla BRICS + inter_bonds，引入 k=1 的 pseudo-overlap。

    对每条跨 fragment 键 (a, b, fa, fb)：
      - 选一个端点原子 shared_atom（规则：按 fragment id 顺序选，保证确定性）；
      - 把 shared_atom 加入 fa 和 fb 对应的 fragment 原子集合中。

    这样每条 BRICS 跨 fragment 键 => 两个 fragment 之间有 1 个共享原子。
    """
    groups_overlap: List[Set[int]] = [set(g) for g in groups_vanilla]

    for a, b, fa, fb in inter_bonds:
        if fa == fb:
            continue

        # 稳定规则：fragment id 较小的那一端作为 shared_atom 的“基准”
        shared_atom = a if fa < fb else b

        if 0 <= fa < len(groups_overlap):
            groups_overlap[fa].add(shared_atom)
        if 0 <= fb < len(groups_overlap):
            groups_overlap[fb].add(shared_atom)

    return groups_overlap


# ---------------------------------------------------------------------------
# BricsFragmentizer：实现 BaseFragmentizer 接口
# ---------------------------------------------------------------------------

class BricsFragmentizer(BaseFragmentizer):
    """
    Fragmentizer implementing RDKit BRICS rules.

    mode:
      - "vanilla" : 原始 BRICS partition（fragment 间不重叠）
      - "overlap" : 在每条跨 fragment 键处引入 overlap=1（适配 CGMNet 的 fragment graph 构图）

    仍然依赖 vocab.txt 来：
      - 把 fragment SMILES 映射到 vocab id
      - 记录频次（可选）
    """

    def __init__(self, vocab_path: str, mode: str = "overlap"):
        mode = mode.lower()
        if mode not in ("vanilla", "overlap"):
            raise ValueError(f"BricsFragmentizer mode must be 'vanilla' or 'overlap', got {mode}")
        self.mode = mode
        self.vocab_path = vocab_path

        # 读取 vocab（格式与 DoveFragmentizer 保持一致）
        with open(vocab_path, "r") as f:
            lines = f.read().strip().splitlines()

        self.smi_to_id: Dict[str, int] = {}
        self.id_to_smi: List[str] = []
        self.vocab_freq: Dict[str, int] = {}

        # 第一行是 JSON header，从第二行开始
        for line in lines[1:]:
            if not line.strip():
                continue
            smi, _, freq = line.strip().split("\t")
            if smi not in self.smi_to_id:
                self.smi_to_id[smi] = len(self.id_to_smi)
                self.id_to_smi.append(smi)
            self.vocab_freq[smi] = int(freq)

        # UNK token
        self.unk_token = "<unk>"
        if self.unk_token not in self.smi_to_id:
            self.smi_to_id[self.unk_token] = len(self.id_to_smi)
            self.id_to_smi.append(self.unk_token)
        self.unk_id = self.smi_to_id[self.unk_token]

    def get_smi_id(self, smi: str) -> int:
        return self.smi_to_id.get(smi, self.unk_id)

    def tokenize(
        self,
        mol: Chem.Mol,
    ) -> Tuple[List[Set[int]], List[int], List[List[int]]]:
        """
        对 RDKit Mol 做 BRICS partition，返回：

        Returns
        -------
        group_atom_indices : list[set[int]]
            每个 fragment 的原子 index 集合（在 mol 上）。
        fragment_vocab_ids : list[int]
            每个 fragment 的主 vocab id。
        final_details : list[list[int]]
            每个 fragment 的“层次 id 列表”（BRICS 没有层次，这里简单设为 [主 id]）。
        """
        groups_vanilla = brics_partition_atom_sets(mol)
        if not groups_vanilla:
            return [], [], []

        inter_bonds = compute_inter_bonds_from_partition(mol, groups_vanilla)

        if self.mode == "overlap":
            groups = apply_overlap_k1(groups_vanilla, inter_bonds)
        else:
            groups = groups_vanilla

        fragment_vocab_ids: List[int] = []
        final_details: List[List[int]] = []

        for atoms in groups:
            atom_list = list(atoms)
            if not atom_list:
                fragment_vocab_ids.append(self.unk_id)
                final_details.append([self.unk_id])
                continue

            submol = get_submol(mol, atom_list)
            smi = mol_to_smi(submol)
            token_id = self.get_smi_id(smi)
            fragment_vocab_ids.append(token_id)
            # BRICS 没有 merge history，这里用 [主 id] 作为占位
            final_details.append([token_id])

        return groups, fragment_vocab_ids, final_details


# ---------------------------------------------------------------------------
# 来自 r-BRICS 论文实现的 environment 定义
environs = {
    'L1':'[C;D3]([#0,#6,#7,#8])(=O)',
    'L3':'[O;D2]-;!@[#0,#6,#1]',
    'L4':'[C;!D1;!$(C=*)]-;!@[#6]',
    'L5':'[N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)]',
    'L51':'[N;!R;!D1;$(N(!@[N,O]))]',
    'L6':'[C;D3;!R](=O)-;!@[#0,#6,#7,#8]',
    'L7a':'[C;D2,D3]-[#6]',
    'L7b':'[C;D2,D3]-[#6]',
    'L8':'[C;!R;!D1;!$(C!-*);!$(C([H])([H])([H]))]',
    'L81':'[C;!R;!D1;$(C(-[C,N,O,S])(=[N,S]))]',
    'L9':'[RN,n;+0;$([RN,n](@[RC,RN,RO,RS,c,n,o,s])@[RC,RN,RO,RS,c,n,o,s])]',
    'L10':'[N;R;$(N(@C(=O))@[C,N,O,S])]',
    'L11':'[S;D2](-;!@[#0,#6])',
    'L12':'[S;D4]([#6,#0])(=O)(=O)',
    'L12b':'[S;D4;!R](!@O)(!@O)',
    'L13':'[C;$(C(-;@[C,N,O,S])-;@[N,O,S])]',
    'L14':'[c;$(c(:[c,n,o,s]):[n,o,s])]',
    'L14b':'[RC;$([RC](@[RC,RN,RO,RS])@[RN,RO,RS])]',
    'L15':'[C;$(C(-;@C)-;@C)]',
    'L16':'[c;$(c(:c):c)]',
    'L16b':'[RC;$([RC](@[RC])@[RC])]',
    # New environments for chains and rings
    'L17':'[C](-C)(-C)(-C)',
    'L18':'[R!#1;x3]',
    'L19':'[R!#1;x2]',
    'L182':'[R!#1;x3]',
    'L192':'[R!#1;x2]',
    'L20':'[CH2][CH2][CH2][CH3,RC,c,$(C(~[!#6]))]',
    'L21':'[CH2][CH2][CH2][CH2][CH3,RC,c,$(C(~[!#6]))]',
    'L22':'[CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH3,RC,c,$(C(~[!#6]))]',
    'L23':'[CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH3,RC,c,$(C(~[!#6]))]',
    'L30':'[C;D2]([#0,#6,#7,#8,#16])(#[N,C])'
}

reactionDefs = (
    [('1','3','-'), ('1','5','-'), ('1','10','-')],  # L1
    [('30','30','-'), ('30','4','-'), ('30','5','-'), ('30','51','-'),
     ('30','6','-'), ('30','81','-'), ('30','9','-'), ('30','10','-'),
     ('30','11','-'), ('30','12','-'), ('30','12b','-'), ('30','13','-'),
     ('30','14','-'), ('30','14b','-'), ('30','15','-'), ('30','16','-'),
     ('30','16b','-')],  # L30
    [('3','4','-'), ('3','13','-'), ('3','14','-'), ('3','15','-'), ('3','16','-')],  # L3
    [('4','5','-'), ('4','11','-')],  # L4
    [('5','12','-'), ('5','14','-'), ('5','16','-'), ('5','13','-'), ('5','15','-')],  # L5
    [('51','1','-'), ('51','4','-'), ('51','12','-'), ('51','12b','-'),
     ('51','14','-'), ('51','16','-'), ('51','13','-'), ('51','15','-')],  # L51
    [('6','13','-'), ('6','14','-'), ('6','15','-'), ('6','16','-')],  # L6
    [('7a','7b','=')],  # L7
    [('8','9','-'), ('8','10','-'), ('8','13','-'), ('8','14','-'),
     ('8','15','-'), ('8','16','-')],  # L8
    [('81','8','-'), ('81','9','-'), ('81','10','-'), ('81','13','-'),
     ('81','14','-'), ('81','15','-'), ('81','16','-')],  # L81
    [('9','13','-'), ('9','14','-'), ('9','15','-'), ('9','16','-')],  # L9
    [('10','13','-'), ('10','14','-'), ('10','15','-'), ('10','16','-')],  # L10
    [('11','13','-'), ('11','14','-'), ('11','15','-'), ('11','16','-')],  # L11
    [('12b','12b','-'), ('12b','5','-'), ('12b','4','-'), ('12b','13','-'),
     ('12b','14','-'), ('12b','15','-'), ('12b','16','-')],  # L12b
    [('13','14','-;@,!@'), ('13','15','-;!@'), ('13','16','-;@,!@')],  # L13
    [('14','14','-;@,!@'), ('14','15','-;@,!@'), ('14','16','-;@,!@')],  # L14
    [('3','14b','-'), ('5','14b','-'), ('51','14b','-'), ('6','14b','-'),
     ('8','14b','-'), ('81','14b','-'), ('9','14b','-'), ('10','14b','-'),
     ('11','14b','-'), ('12b','14b','-'), ('13','14b','-'), ('14','14b','-'),
     ('14b','14b','-'), ('14b','16','-'), ('14b','16b','-'), ('14b','15','-'),
     ('14b','17','-')],  # L14b
    [('15','16','-;@,!@')],  # L15
    [('16','16','-;@,!@')],  # L16
    [('3','16b','-'), ('5','16b','-'), ('51','16b','-'), ('6','16b','-'),
     ('8','16b','-'), ('81','16b','-'), ('9','16b','-'), ('10','16b','-'),
     ('11','16b','-'), ('12b','16b','-'), ('13','16b','-'), ('15','16b','-'),
     ('16','16b','-'), ('16b','16b','-'), ('17','16b','-')],  # L16b
    [('17','17','-'), ('17','16','-'), ('17','15','-'), ('17','14','-'),
     ('17','13','-'), ('17','12b','-'), ('17','12','-'), ('17','11','-'),
     ('17','10','-'), ('17','9','-'), ('17','8','-'), ('17','81','-'),
     ('17','51','-'), ('17','5','-')],  # L17
    [('18','19','-;@'), ('182','192','=;@')],  # L18 (Rings)
    [('20','21','-')],  # L20 (Chains)
    [('22','23','-')]   # L22 (Chains)
)

def init_rbrics_reactions(environs: dict, reactionDefs) -> tuple[dict, list]:
    """初始化 r-BRICS 的 environment / bond 匹配器（基本照搬沙盒实现）"""
    smartsGps = copy.deepcopy(reactionDefs)
    for gp in smartsGps:
        for j, defn in enumerate(gp):
            g1, g2, bnd = defn
            r1 = environs['L' + g1]
            r2 = environs['L' + g2]
            g1n = re.sub('[a-z,A-Z]', '', g1)
            g2n = re.sub('[a-z,A-Z]', '', g2)
            if "@" not in bnd:
                sma = '[$(%s):1]%s;!@[$(%s):2]>>[%s*]-[*:1].[%s*]-[*:2]' % (r1, bnd, r2, g1n, g2n)
            else:
                sma = '[$(%s):1]%s[$(%s):2]>>[%s*]-[*:1].[%s*]-[*:2]' % (r1, bnd, r2, g1n, g2n)
            gp[j] = sma

    environMatchers = {}
    for env, sma in environs.items():
        environMatchers[env] = Chem.MolFromSmarts(sma)

    bondMatchers = []
    for compats in reactionDefs:
        tmp = []
        for i1, i2, bType in compats:
            e1 = environs['L' + i1]
            e2 = environs['L' + i2]
            if "@" in bType:
                patt = '[$(%s)]%s[$(%s)]' % (e1, bType, e2)
            else:
                patt = '[$(%s)]%s;!@[$(%s)]' % (e1, bType, e2)
            patt = Chem.MolFromSmarts(patt)
            tmp.append((i1, i2, bType, patt))
        bondMatchers.append(tmp)

    return environMatchers, bondMatchers

# 全局 matcher（与沙盒一致，模块 import 时初始化一次）
environMatchers, bondMatchers = init_rbrics_reactions(environs, reactionDefs)


def FindrBRICSBonds(mol: Chem.Mol, randomizeOrder: bool = False):
    """查找 r-BRICS 切断的键，照搬沙盒 FindrBRICSBonds。"""
    letter = re.compile('[a-z,A-Z]')
    indices = list(range(len(bondMatchers)))
    bondsDone = set()
    if randomizeOrder:
        import random
        random.shuffle(indices)

    envMatches = {}
    for env, patt in environMatchers.items():
        envMatches[env] = mol.HasSubstructMatch(patt)

    for gpIdx in indices:
        compats = bondMatchers[gpIdx]
        for i1, i2, bType, patt in compats:
            if not envMatches['L' + i1] or not envMatches['L' + i2]:
                continue
            matches = mol.GetSubstructMatches(patt)
            i1n = letter.sub('', i1)
            i2n = letter.sub('', i2)
            for match in matches:
                if match not in bondsDone and (match[1], match[0]) not in bondsDone:
                    bondsDone.add(match)
                    yield ( (match[0], match[1]), (i1n, i2n) )


def BreakrBRICSBonds(mol: Chem.Mol, bonds, sanitize: bool = True) -> Chem.Mol:
    """
    在指定键上断开，并插入 dummy atom（沙盒 BreakrBRICSBonds 的移植版）。
    要求 bonds 由 FindrBRICSBonds 提供。
    """
    eMol = Chem.EditableMol(mol)
    nAts = mol.GetNumAtoms()
    dummyPositions = []

    for indices, dummyTypes in bonds:
        ia, ib = indices
        obond = mol.GetBondBetweenAtoms(ia, ib)
        if obond is None:
            continue
        bondType = obond.GetBondType()
        eMol.RemoveBond(ia, ib)

        da, db = dummyTypes
        atoma = Chem.Atom(0)
        atoma.SetIsotope(int(da))
        atoma.SetNoImplicit(True)
        idxa = nAts
        nAts += 1
        eMol.AddAtom(atoma)
        eMol.AddBond(ia, idxa, bondType)

        atomb = Chem.Atom(0)
        atomb.SetIsotope(int(db))
        atomb.SetNoImplicit(True)
        idxb = nAts
        nAts += 1
        eMol.AddAtom(atomb)
        eMol.AddBond(ib, idxb, bondType)

        if mol.GetNumConformers():
            dummyPositions.append((idxa, ib))
            dummyPositions.append((idxb, ia))

    res = eMol.GetMol()
    if sanitize:
        Chem.SanitizeMol(res)

    if mol.GetNumConformers():
        for conf in mol.GetConformers():
            resConf = res.GetConformer(conf.GetId())
            for ia, pa in dummyPositions:
                resConf.SetAtomPosition(ia, conf.GetAtomPosition(pa))
    return res


def rbrics_partition_atom_sets(mol: Chem.Mol) -> List[Set[int]]:
    """
    r-BRICS 分割：输入假定为“无显式氢”的 RDKit Mol，
    输出每个 fragment 的原子 index 集合（对应原始 mol 的 index）。

    完全保持沙盒版 rbrics_partition 的语义：
      - 如果没有可切断的键，就返回整个分子一个 fragment。
      - 使用 Kekulize(clearAromaticFlags=True) 来打破芳环。
    """
    # 在拷贝上操作，避免修改原始 mol（featurizer 还要用）
    mol_work = Chem.Mol(mol)
    for atom in mol_work.GetAtoms():
        atom.SetIntProp("orig_idx", atom.GetIdx())

    # r-BRICS 关键：Kekulize 以便断环
    Chem.Kekulize(mol_work, clearAromaticFlags=True)

    bonds_to_cut = list(FindrBRICSBonds(mol_work))
    if not bonds_to_cut:
        # 与沙盒一致：没有切点时返回整个分子
        return [set(range(mol_work.GetNumAtoms()))]

    fragmented = BreakrBRICSBonds(mol_work, bonds_to_cut)
    frag_tuples = Chem.GetMolFrags(fragmented, asMols=False)

    group_atom_indices: List[Set[int]] = []
    for frag_indices in frag_tuples:
        cur: Set[int] = set()
        for idx in frag_indices:
            atom = fragmented.GetAtomWithIdx(idx)
            if atom.HasProp("orig_idx"):
                cur.add(atom.GetIntProp("orig_idx"))
        if cur:
            group_atom_indices.append(cur)

    if not group_atom_indices:
        # 兜底：如果因为某些极端情况失败，仍然返回一个整体 fragment
        return [set(range(mol_work.GetNumAtoms()))]

    return group_atom_indices

# ---------------------------------------------------------------------------
# RBricsFragmentizer：实现 BaseFragmentizer 接口（r-BRICS）
# ---------------------------------------------------------------------------

class RBricsFragmentizer(BaseFragmentizer):
    """
    Fragmentizer implementing r-BRICS rules (ring-breaking + chain-breaking).

    mode:
      - "vanilla" : 原始 r-BRICS partition（fragment 间不重叠）
      - "overlap" : 在每条跨 fragment 键处引入 overlap=1（适配 CGMNet 的 fragment graph）

    同样依赖 vocab.txt：
      - fragment SMILES -> vocab id
      - 频次作为可选信息
    """

    def __init__(self, vocab_path: str, mode: str = "overlap"):
        mode = mode.lower()
        if mode not in ("vanilla", "overlap"):
            raise ValueError(f"RBricsFragmentizer mode must be 'vanilla' or 'overlap', got {mode}")
        self.mode = mode
        self.vocab_path = vocab_path

        with open(vocab_path, "r") as f:
            lines = f.read().strip().splitlines()

        self.smi_to_id: Dict[str, int] = {}
        self.id_to_smi: List[str] = []
        self.vocab_freq: Dict[str, int] = {}

        # 第一行是 JSON header，从第二行开始
        for line in lines[1:]:
            if not line.strip():
                continue
            smi, _, freq = line.strip().split("\t")
            if smi not in self.smi_to_id:
                self.smi_to_id[smi] = len(self.id_to_smi)
                self.id_to_smi.append(smi)
            self.vocab_freq[smi] = int(freq)

        # UNK token
        self.unk_token = "<unk>"
        if self.unk_token not in self.smi_to_id:
            self.smi_to_id[self.unk_token] = len(self.id_to_smi)
            self.id_to_smi.append(self.unk_token)
        self.unk_id = self.smi_to_id[self.unk_token]

    def get_smi_id(self, smi: str) -> int:
        return self.smi_to_id.get(smi, self.unk_id)

    def tokenize(
        self,
        mol: Chem.Mol,
    ) -> Tuple[List[Set[int]], List[int], List[List[int]]]:
        """
        对 RDKit Mol 做 r-BRICS partition，返回：

        group_atom_indices : list[set[int]]
            每个 fragment 的原子 index 集合（在原始 mol 上）。
        fragment_vocab_ids : list[int]
            每个 fragment 的主 vocab id。
        final_details : list[list[int]]
            每个 fragment 的“层次 id 列表”（r-BRICS 没有合并历史，这里用 [主 id] 占位）。
        """
        groups_vanilla = rbrics_partition_atom_sets(mol)
        if not groups_vanilla:
            return [], [], []

        # 这里直接重用 BRICS 的 inter_bonds 计算和 overlap 逻辑
        inter_bonds = compute_inter_bonds_from_partition(mol, groups_vanilla)

        if self.mode == "overlap":
            groups = apply_overlap_k1(groups_vanilla, inter_bonds)
        else:
            groups = groups_vanilla

        fragment_vocab_ids: List[int] = []
        final_details: List[List[int]] = []

        for atoms in groups:
            atom_list = list(atoms)
            if not atom_list:
                fragment_vocab_ids.append(self.unk_id)
                final_details.append([self.unk_id])
                continue

            submol = get_submol(mol, atom_list)
            smi = mol_to_smi(submol)
            token_id = self.get_smi_id(smi)
            fragment_vocab_ids.append(token_id)
            final_details.append([token_id])

        return groups, fragment_vocab_ids, final_details

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# RECAP helpers: vanilla partition + overlap=k1
# ---------------------------------------------------------------------------

# RECAP SMARTS extracted and adapted from RDKit / Lewell 1998
RECAP_SMARTS = [
    # 1. Amide (酰胺): C(=O)-N
    "[C;!$(C([#7])[#7])]!@[#7;+0;!D1]",

    # 2. Ester (酯): C(=O)-O
    "[C](=!@[O])!@[O;+0]",

    # 3. Urea (脲): N-C(=O)-N
    "[#7;+0;D2,D3]!@[C;!$(C=[!O])]",

    # 4. Amines (胺): C-N（排除酰胺等）
    "[N;!D1;+0;!$(N-C=[#7,#8,#15,#16])]!@[*]",

    # 5. Cyclic Amines (环胺)
    "[#7;R;D3;+0]!@[*]",

    # 6. Ether (醚): C-O-C
    "[#6]!@[O;+0]!@[#6]",

    # 7. Olefin (烯烃): C=C
    "[C]=!@[C]",

    # 8. Aromatic N - Aliphatic C
    "[n;+0]!@[C]",

    # 9. Lactam N - Aliphatic C
    "[O]=[C]-@[N;+0]!@[C]",

    # 10. Aromatic C - Aromatic C
    "[c]!@[c]",

    # 11. Aromatic N - Aromatic C
    "[n;+0]!@[c]",

    # 12. Sulphonamide: N-S(=O)(=O)
    "[#7;+0;D2,D3]!@[S](=[O])=[O]",
]


def break_recap_bonds(mol: Chem.Mol) -> Chem.Mol:
    """
    按 RECAP 规则切键：
      - 用一组 SMARTS 定义可切键的环境；
      - 只切 non-ring 键（!@）。
    """
    bonds_to_cut: set[int] = set()

    for pattern in RECAP_SMARTS:
        smarts = Chem.MolFromSmarts(pattern)
        if not smarts:
            continue

        matches = mol.GetSubstructMatches(smarts)
        for match in matches:
            # 枚举匹配里的原子对，找出真实存在的键并且是非环键
            for i in range(len(match)):
                for j in range(i + 1, len(match)):
                    u, v = match[i], match[j]
                    bond = mol.GetBondBetweenAtoms(u, v)
                    if bond is None:
                        continue
                    if not bond.IsInRing():
                        bonds_to_cut.add(bond.GetIdx())

    if not bonds_to_cut:
        # 没有可切的键：直接返回拷贝
        return Chem.Mol(mol)

    # 根据键 index 断键，dummyLabels 用占位 (0,0)，只用 orig_idx 回溯原子
    fragmented = Chem.FragmentOnBonds(
        mol,
        list(bonds_to_cut),
        dummyLabels=[(0, 0)] * len(bonds_to_cut),
    )
    return fragmented


def recap_partition_atom_sets(mol: Chem.Mol) -> List[Set[int]]:
    """
    RECAP partition：输入为“无显式氢”的 RDKit Mol，
    输出每个 fragment 在原始 mol 上的原子 index 集合。

    语义与沙盒版本 recap_partition 一致：
      - 先给每个 atom 打 orig_idx；
      - RECAP 切键；
      - 按 frag 拿回 orig_idx；
      - 若最终为空，则退化为整个分子一个 fragment。
    """
    # 在拷贝上操作，避免污染原分子
    mol_work = Chem.Mol(mol)
    for atom in mol_work.GetAtoms():
        atom.SetIntProp("orig_idx", atom.GetIdx())

    fragmented = break_recap_bonds(mol_work)
    frag_tuples = Chem.GetMolFrags(fragmented, asMols=False)

    groups: List[Set[int]] = []
    for frag_indices in frag_tuples:
        cur: Set[int] = set()
        for idx in frag_indices:
            atom = fragmented.GetAtomWithIdx(idx)
            if atom.HasProp("orig_idx"):
                cur.add(atom.GetIntProp("orig_idx"))
        if cur:
            groups.append(cur)

    if not groups:
        # 极端兜底：视作单一 fragment
        return [set(range(mol_work.GetNumAtoms()))]

    return groups
# ---------------------------------------------------------------------------
# RecapFragmentizer：实现 BaseFragmentizer 接口（RECAP）
# ---------------------------------------------------------------------------

class RecapFragmentizer(BaseFragmentizer):
    """
    Fragmentizer implementing RECAP rules.

    mode:
      - "vanilla" : 原始 RECAP partition（fragment 间不重叠）
      - "overlap" : 在每条跨 fragment 键处引入 overlap=1（适配 CGMNet 的 fragment graph）

    同样依赖 vocab.txt：
      - fragment SMILES -> vocab id
      - 频次作为可选信息
    """

    def __init__(self, vocab_path: str, mode: str = "overlap"):
        mode = mode.lower()
        if mode not in ("vanilla", "overlap"):
            raise ValueError(f"RecapFragmentizer mode must be 'vanilla' or 'overlap', got {mode}")
        self.mode = mode
        self.vocab_path = vocab_path

        with open(vocab_path, "r") as f:
            lines = f.read().strip().splitlines()

        self.smi_to_id: Dict[str, int] = {}
        self.id_to_smi: List[str] = []
        self.vocab_freq: Dict[str, int] = {}

        # 第一行是 JSON header，从第二行开始
        for line in lines[1:]:
            if not line.strip():
                continue
            smi, _, freq = line.strip().split("\t")
            if smi not in self.smi_to_id:
                self.smi_to_id[smi] = len(self.id_to_smi)
                self.id_to_smi.append(smi)
            self.vocab_freq[smi] = int(freq)

        # UNK token
        self.unk_token = "<unk>"
        if self.unk_token not in self.smi_to_id:
            self.smi_to_id[self.unk_token] = len(self.id_to_smi)
            self.id_to_smi.append(self.unk_token)
        self.unk_id = self.smi_to_id[self.unk_token]

    def get_smi_id(self, smi: str) -> int:
        return self.smi_to_id.get(smi, self.unk_id)

    def tokenize(
        self,
        mol: Chem.Mol,
    ) -> Tuple[List[Set[int]], List[int], List[List[int]]]:
        """
        对 RDKit Mol 做 RECAP partition，返回：

        group_atom_indices : list[set[int]]
            每个 fragment 的原子 index 集合（在原始 mol 上）。
        fragment_vocab_ids : list[int]
            每个 fragment 的主 vocab id。
        final_details : list[list[int]]
            每个 fragment 的“层次 id 列表”（RECAP 没有合并历史，这里用 [主 id] 占位）。
        """
        groups_vanilla = recap_partition_atom_sets(mol)
        if not groups_vanilla:
            return [], [], []

        # 重用 BRICS 的 inter_bonds 计算 + overlap 逻辑
        inter_bonds = compute_inter_bonds_from_partition(mol, groups_vanilla)

        if self.mode == "overlap":
            groups = apply_overlap_k1(groups_vanilla, inter_bonds)
        else:
            groups = groups_vanilla

        fragment_vocab_ids: List[int] = []
        final_details: List[List[int]] = []

        for atoms in groups:
            atom_list = list(atoms)
            if not atom_list:
                fragment_vocab_ids.append(self.unk_id)
                final_details.append([self.unk_id])
                continue

            submol = get_submol(mol, atom_list)
            smi = mol_to_smi(submol)
            token_id = self.get_smi_id(smi)
            fragment_vocab_ids.append(token_id)
            final_details.append([token_id])

        return groups, fragment_vocab_ids, final_details

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ReLMole helpers：环 + 功能团 + linker 分割
# ---------------------------------------------------------------------------

# 来自你 relmole_pdb_san.py 的核心模式
RELMOLE_PATT_SMARTS = {
    "HETEROATOM": "[!#6]",
    "DOUBLE_TRIPLE_BOND": "*=,#*",
    "ACETAL": "[CX4]([O,N,S])[O,N,S]",
}
RELMOLE_PATTS = {k: Chem.MolFromSmarts(v) for k, v in RELMOLE_PATT_SMARTS.items()}


def _relmole_get_fragments(mol: Chem.Mol) -> List[Set[int]]:
    """
    ReLMole 原始分割逻辑（vanilla 模式）：

      1) 先把所有环合并成 ring systems；
      2) 再根据 hetero / double/triple / acetal 模式找功能团；
      3) 最后用 bond 连起来，剩下的当作 linker/skeleton。

    返回：每个 fragment 的原子 index 集合（在 mol 上）。
    """
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return []

    # --- 1. 合并 fused rings ---
    try:
        rings = [set(x) for x in Chem.GetSymmSSSR(mol)]
    except Exception:
        rings = []

    # 把共享原子数 > 2 的环合并成 ring system
    flag = True
    while flag:
        flag = False
        for i in range(len(rings)):
            if not rings[i]:
                continue
            for j in range(i + 1, len(rings)):
                if not rings[j]:
                    continue
                shared = rings[i] & rings[j]
                if len(shared) > 2:
                    rings[i].update(rings[j])
                    rings[j] = set()
                    flag = True
    rings = [r for r in rings if r]

    # --- 2. 找功能团“核心原子” ---
    marks: Set[int] = set()
    for patt in RELMOLE_PATTS.values():
        if patt is None:
            continue
        for sub in mol.GetSubstructMatches(patt):
            marks.update(sub)

    # --- 3. 沿着非环键把功能团和 linker 串起来 ---
    fgs: List[Set[int]] = []
    atom2fg: List[List[int]] = [[] for _ in range(num_atoms)]

    # 先把每个被标记的原子初始化为一个 fg
    for atom_idx in marks:
        fg_id = len(fgs)
        fgs.append({atom_idx})
        atom2fg[atom_idx] = [fg_id]

    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue

        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()

        in1 = a1 in marks
        in2 = a2 in marks

        if in1 and in2:
            # 两个都在 marks：把两个 fg merge
            fg1 = atom2fg[a1][0]
            fg2 = atom2fg[a2][0]
            if fg1 != fg2:
                fgs[fg1].update(fgs[fg2])
                fgs[fg2] = set()
                for at in fgs[fg1]:
                    atom2fg[at] = [fg1]
        elif in1:
            # a1 是功能团核心，顺着键把 a2 吸进去
            fg = atom2fg[a1][0]
            fgs[fg].add(a2)
            atom2fg[a2].append(fg)
        elif in2:
            fg = atom2fg[a2][0]
            fgs[fg].add(a1)
            atom2fg[a1].append(fg)
        else:
            # 两个都不是 marks：作为 linker 片段
            fg_id = len(fgs)
            fgs.append({a1, a2})
            atom2fg[a1].append(fg_id)
            atom2fg[a2].append(fg_id)

    # --- 4. 汇总：rings 在前，fg/linker 在后 ---
    final_groups: List[Set[int]] = []

    # 先放 ring systems
    for r in rings:
        if r:
            final_groups.append(set(r))

    # 再放 fgs；去掉 (单原子且在环里) 这种已被 rings 覆盖的情况
    for fg in fgs:
        if not fg:
            continue
        if len(fg) == 1:
            only = next(iter(fg))
            if mol.GetAtomWithIdx(only).IsInRing():
                continue
        final_groups.append(set(fg))

    # 极端兜底：如果一个都没分出来，就把整个分子当作一个 fragment
    if not final_groups:
        final_groups = [set(range(num_atoms))]

    return final_groups


def relmole_partition_atom_sets(mol: Chem.Mol) -> List[Set[int]]:
    """
    对无显式氢的 RDKit Mol 做 ReLMole partition，返回每个 fragment 的原子 index 集合。

    注意：不做任何 overlap，纯 vanilla ReLMole。
    """
    try:
        return _relmole_get_fragments(mol)
    except Exception:
        # 防止奇怪分子导致 GetSymmSSSR 等崩掉
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return []
        return [set(range(num_atoms))]


def relmole_apply_overlap(
    mol: Chem.Mol,
    groups_vanilla: List[Set[int]],
) -> List[Set[int]]:
    """
    ReLMole overlap 模式：

      - 输入 vanilla 分组；
      - 对每对 fragment (i, j)，如果它们之间存在一条跨 fragment 键，
        则从该键上选一对原子 a(i->j), b(j->i)，互相加入对方 fragment，
        从而引入 1-atom overlap。

    语义基本对应你沙盒里的 find_overlapping_connections + apply_overlap_logic。
    """
    n = len(groups_vanilla)
    if n <= 1:
        return [set(g) for g in groups_vanilla]

    groups = [set(g) for g in groups_vanilla]

    # 预先建一个 atom -> fragment ids 的索引，方便判断 fa / fb
    atom_to_frags: Dict[int, List[int]] = {}
    for fid, g in enumerate(groups_vanilla):
        for a in g:
            atom_to_frags.setdefault(a, []).append(fid)

    # 遍历所有非环键，找跨 fragment 的 (fa, fb)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        fa_list = atom_to_frags.get(a1, [])
        fb_list = atom_to_frags.get(a2, [])
        if not fa_list or not fb_list:
            continue

        # 理论上 ReLMole 不应该出现一个原子属于多个 vanilla fragment，
        # 这里还是稍微兜底一下：取第一个。
        fa = fa_list[0]
        fb = fb_list[0]
        if fa == fb:
            continue

        i, j = (fa, fb) if fa < fb else (fb, fa)
        gi, gj = groups[i], groups[j]

        # 只有在目前还是不相交的时候才引入 overlap
        if gi.isdisjoint(gj):
            # 让这两个 fragment 共享这条键的两端原子
            gi.add(a1)
            gi.add(a2)
            gj.add(a1)
            gj.add(a2)

    return groups


# ---------------------------------------------------------------------------
# ReLMoleFragmentizer：实现 BaseFragmentizer 接口（ReLMole）
# ---------------------------------------------------------------------------

class ReLMoleFragmentizer(BaseFragmentizer):
    """
    Fragmentizer implementing ReLMole rules (rings + FGs + linker).

    mode:
      - "vanilla" : 纯 ReLMole partition（fragment 间不重叠）
      - "overlap" : 根据键邻接关系引入 1-atom overlap
    """

    def __init__(self, vocab_path: str, mode: str = "overlap"):
        mode = mode.lower()
        if mode not in ("vanilla", "overlap"):
            raise ValueError(f"ReLMoleFragmentizer mode must be 'vanilla' or 'overlap', got {mode}")
        self.mode = mode
        self.vocab_path = vocab_path

        with open(vocab_path, "r") as f:
            lines = f.read().strip().splitlines()

        self.smi_to_id: Dict[str, int] = {}
        self.id_to_smi: List[str] = []
        self.vocab_freq: Dict[str, int] = {}

        # 第一行是 JSON header，从第二行开始
        for line in lines[1:]:
            if not line.strip():
                continue
            smi, _, freq = line.strip().split("\t")
            if smi not in self.smi_to_id:
                self.smi_to_id[smi] = len(self.id_to_smi)
                self.id_to_smi.append(smi)
            self.vocab_freq[smi] = int(freq)

        # UNK token
        self.unk_token = "<unk>"
        if self.unk_token not in self.smi_to_id:
            self.smi_to_id[self.unk_token] = len(self.id_to_smi)
            self.id_to_smi.append(self.unk_token)
        self.unk_id = self.smi_to_id[self.unk_token]

    def get_smi_id(self, smi: str) -> int:
        return self.smi_to_id.get(smi, self.unk_id)

    def tokenize(
        self,
        mol: Chem.Mol,
    ) -> Tuple[List[Set[int]], List[int], List[List[int]]]:
        """
        对 RDKit Mol 做 ReLMole partition，返回：

        group_atom_indices : list[set[int]]
            每个 fragment 的原子 index 集合（在原始 mol 上）。
        fragment_vocab_ids : list[int]
            每个 fragment 的主 vocab id。
        final_details : list[list[int]]
            每个 fragment 的“层次 id 列表”（ReLMole 没有合并历史，这里用 [主 id] 占位）。
        """
        groups_vanilla = relmole_partition_atom_sets(mol)
        if not groups_vanilla:
            return [], [], []

        if self.mode == "overlap":
            groups = relmole_apply_overlap(mol, groups_vanilla)
        else:
            groups = groups_vanilla

        fragment_vocab_ids: List[int] = []
        final_details: List[List[int]] = []

        for atoms in groups:
            atom_list = list(atoms)
            if not atom_list:
                fragment_vocab_ids.append(self.unk_id)
                final_details.append([self.unk_id])
                continue

            submol = get_submol(mol, atom_list)
            smi = mol_to_smi(submol)
            token_id = self.get_smi_id(smi)
            fragment_vocab_ids.append(token_id)
            final_details.append([token_id])

        return groups, fragment_vocab_ids, final_details


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# JT-VAE Junction Tree decomposition helpers
# ---------------------------------------------------------------------------

def jt_tree_decomp(mol: Chem.Mol) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    """
    JT-VAE 风格的 junction tree 分解 (Jin et al., ICML 2018).

    返回：
      cliques : list[list[int]]   每个 cluster 对应一组原子 index
      edges   : list[(int,int)]   cluster 之间的树结构边（这里只用于调试/可视化）
    """
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    # 1. 初始化 cliques：所有非环键 + 所有环
    cliques: List[List[int]] = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            cliques.append([a1, a2])  # 非环键视作 clique

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques.extend(ssr)  # 环也是 clique

    # atom -> 属于哪些 clique
    nei_list: List[List[int]] = [[] for _ in range(n_atoms)]
    for i, atoms in enumerate(cliques):
        for a in atoms:
            nei_list[a].append(i)

    # 2. 按 Jin 论文逻辑建立 clique 间的加权边
    MST_MAX_WEIGHT = 100
    edges_w = defaultdict(int)  # (c1,c2) -> weight

    for atom in range(n_atoms):
        if len(nei_list[atom]) <= 1:
            continue

        cnei = nei_list[atom]
        bonds = [c for c in cnei if len(cliques[c]) == 2]
        rings = [c for c in cnei if len(cliques[c]) > 4]

        # 复杂交叉，用“虚拟”单原子节点
        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2):
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges_w[(c1, c2)] = 1
        elif len(rings) > 2:
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges_w[(c1, c2)] = MST_MAX_WEIGHT - 1
        else:
            # 一般情况：按 clique 间交集大小加权
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1, c2 = cnei[i], cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges_w[(c1, c2)] < len(inter):
                        edges_w[(c1, c2)] = len(inter)

    if not edges_w:
        return cliques, []

    # 3. 用 NetworkX 做 maximum spanning tree（等价于原 SciPy 版本的 trick）
    G = nx.Graph()
    for (u, v), w in edges_w.items():
        G.add_edge(int(u), int(v), weight=float(w))

    T = nx.maximum_spanning_tree(G, weight="weight")
    final_edges: List[Tuple[int, int]] = []
    for u, v in T.edges():
        final_edges.append((int(u), int(v)))

    return cliques, final_edges


def jt_partition_atom_sets(mol: Chem.Mol) -> List[Set[int]]:
    """
    JT 分割：把 cliques 映射成 fragment 原子集合（与 featurizer 约定的 index 对齐）。
    """
    cliques, _ = jt_tree_decomp(mol)
    groups: List[Set[int]] = []
    for atoms in cliques:
        s = set(int(a) for a in atoms)
        if s:
            groups.append(s)
    if not groups:
        # 极端情况下兜底：整体一个 fragment
        return [set(range(mol.GetNumAtoms()))]
    return groups

# ---------------------------------------------------------------------------
# JTFragmentizer：实现 BaseFragmentizer 接口（Junction Tree）
# ---------------------------------------------------------------------------

class JTFragmentizer(BaseFragmentizer):
    """
    Fragmentizer implementing JT-VAE style junction-tree decomposition.

    - fragment = clique（环 / 键 / 单原子 hub）
    - 允许原子在多个 fragment 中重叠（天然 overlap）
    - final_details 用 [主 vocab id] 占位（没有 DOVE 式的 merge history）
    """

    def __init__(self, vocab_path: str):
        self.vocab_path = vocab_path

        with open(vocab_path, "r") as f:
            lines = f.read().strip().splitlines()

        self.smi_to_id: Dict[str, int] = {}
        self.id_to_smi: List[str] = []
        self.vocab_freq: Dict[str, int] = {}

        # 第一行是 JSON header，从第二行开始
        for line in lines[1:]:
            if not line.strip():
                continue
            smi, _, freq = line.strip().split("\t")
            if smi not in self.smi_to_id:
                self.smi_to_id[smi] = len(self.id_to_smi)
                self.id_to_smi.append(smi)
            self.vocab_freq[smi] = int(freq)

        # UNK token
        self.unk_token = "<unk>"
        if self.unk_token not in self.smi_to_id:
            self.smi_to_id[self.unk_token] = len(self.id_to_smi)
            self.id_to_smi.append(self.unk_token)
        self.unk_id = self.smi_to_id[self.unk_token]

    def get_smi_id(self, smi: str) -> int:
        return self.smi_to_id.get(smi, self.unk_id)

    def tokenize(
        self,
        mol: Chem.Mol,
    ) -> Tuple[List[Set[int]], List[int], List[List[int]]]:
        """
        对 RDKit Mol 做 JT 分解，返回：
          - group_atom_indices : 每个 clique 对应的原子 index 集合
          - fragment_vocab_ids : 每个 fragment 的主 vocab id
          - final_details      : 这里简单用 [主 id] 占位
        """
        groups = jt_partition_atom_sets(mol)
        if not groups:
            return [], [], []

        fragment_vocab_ids: List[int] = []
        final_details: List[List[int]] = []

        for atoms in groups:
            atom_list = list(atoms)
            if not atom_list:
                fragment_vocab_ids.append(self.unk_id)
                final_details.append([self.unk_id])
                continue

            submol = get_submol(mol, atom_list)
            smi = mol_to_smi(submol)
            token_id = self.get_smi_id(smi)
            fragment_vocab_ids.append(token_id)
            final_details.append([token_id])

        return groups, fragment_vocab_ids, final_details



# ---------------------------------------------------------------------------
# MacFrag helpers: BRICS-based multi-scale fragments
# ---------------------------------------------------------------------------


def macfrag_get_brics_components(
    mol: Chem.Mol,
) -> tuple[list[Chem.Mol], list[set[int]]]:
    """
    在“无显式氢”的 RDKit Mol 上跑 BRICS，返回：
      - frags_mol_list : 每个 fragment 对应一个带 dummy atom 的 RDKit Mol
      - base_groups    : 每个 fragment 在原分子上的原子 index 集合（orig_idx）

    语义与沙盒版 get_brics_components 一致，但不做 3D。
    """
    # 给原分子每个原子打 orig_idx 标签
    mol_work = Chem.Mol(mol)
    for atom in mol_work.GetAtoms():
        atom.SetIntProp("orig_idx", atom.GetIdx())

    fragmented = BRICS.BreakBRICSBonds(mol_work)
    frags_mol_list = list(Chem.GetMolFrags(fragmented, asMols=True))

    base_groups: list[set[int]] = []
    for fmol in frags_mol_list:
        indices: set[int] = set()
        for atom in fmol.GetAtoms():
            if atom.HasProp("orig_idx"):
                indices.add(atom.GetIntProp("orig_idx"))
        base_groups.append(indices)

    return frags_mol_list, base_groups


def macfrag_analyze_connections(
    frags_mol_list: list[Chem.Mol],
    original_mol: Chem.Mol,
) -> tuple[dict[int, set[int]], dict[tuple[int, int], tuple[int, int]]]:
    """
    建立 MacFrag stitching 所需的信息：

      - adj: fragment-level 邻接表（基于原分子上的键连接）
      - dummy_map: (frag_id, dummy_atom_idx) -> (target_frag_id, target_atom_orig_idx)

    基本照搬沙盒 analyze_macfrag_connections。
    """
    # 先把“原子 -> fragment”映射建好
    atom_to_frag: dict[int, int] = {}
    for fid, fmol in enumerate(frags_mol_list):
        for atom in fmol.GetAtoms():
            if atom.HasProp("orig_idx"):
                atom_to_frag[atom.GetIntProp("orig_idx")] = fid

    adj: dict[int, set[int]] = {i: set() for i in range(len(frags_mol_list))}
    dummy_map: dict[tuple[int, int], tuple[int, int]] = {}

    for fid, fmol in enumerate(frags_mol_list):
        for atom in fmol.GetAtoms():
            if atom.GetAtomicNum() != 0:  # dummy 原子
                continue

            neighbors = atom.GetNeighbors()
            if not neighbors:
                continue
            # dummy 在本 fragment 里只有一个邻居真实原子
            neighbor = neighbors[0]
            if not neighbor.HasProp("orig_idx"):
                continue
            real_oid = neighbor.GetIntProp("orig_idx")

            # 看看 real_oid 在原分子里连到谁
            orig_atom = original_mol.GetAtomWithIdx(real_oid)
            for orig_nb in orig_atom.GetNeighbors():
                nb_oid = orig_nb.GetIdx()
                target_frag = atom_to_frag.get(nb_oid)
                if target_frag is None or target_frag == fid:
                    continue

                # 记录 fragment-level 的邻接关系
                adj[fid].add(target_frag)
                adj[target_frag].add(fid)

                # dummy 对应的连接信息：(本 frag, dummy_idx) -> (目标 frag, 目标原子 orig_idx)
                dummy_map[(fid, atom.GetIdx())] = (target_frag, nb_oid)
                break

    return adj, dummy_map


def macfrag_bfs_enumerate_subgraphs(
    adj: dict[int, set[int]],
    max_k: int,
) -> list[list[int]]:
    """
    在 fragment-level 图上做 BFS 枚举所有 size <= max_k 的联通子图。
    返回：每个子图是一组 fragment id 的有序列表。
    """
    num_nodes = len(adj)
    subgraphs: set[frozenset[int]] = set()
    queue: list[frozenset[int]] = []

    # 所有单点子图
    for i in range(num_nodes):
        sg = frozenset([i])
        subgraphs.add(sg)
        queue.append(sg)

    current_level = queue
    for _ in range(max_k - 1):
        next_level: list[frozenset[int]] = []
        seen_level: set[frozenset[int]] = set()
        for sg in current_level:
            neighbors: set[int] = set()
            for node in sg:
                neighbors.update(adj.get(node, set()))
            neighbors -= sg
            for n in neighbors:
                new_sg = frozenset(set(sg) | {n})
                if new_sg in subgraphs:
                    continue
                subgraphs.add(new_sg)
                if new_sg not in seen_level:
                    next_level.append(new_sg)
                    seen_level.add(new_sg)
        if not next_level:
            break
        current_level = next_level

    # 转成排好序的 list[list[int]]
    return [sorted(list(sg)) for sg in subgraphs]


def stitch_macro_fragment(
    frag_ids: list[int],
    frags_mol_list: list[Chem.Mol],
    dummy_map: dict[tuple[int, int], tuple[int, int]],
) -> str:
    """
    把若干 BRICS fragment “缝合”成一个宏碎片：

      - 内部 dummy（连接在子图内部的）会被跳过，真实原子直接连接；
      - 外部 dummy 保留（依然作为外部 BRICS label）。
    返回宏碎片的 SMILES。
    """
    combined = Chem.RWMol()
    old_to_new_idx: dict[tuple[int, int], int] = {}
    frag_set = set(frag_ids)

    # 1) 复制原子（内部 dummy 不复制）
    for fid in frag_ids:
        mol_f = frags_mol_list[fid]
        for atom in mol_f.GetAtoms():
            is_internal_dummy = False
            if atom.GetAtomicNum() == 0 and (fid, atom.GetIdx()) in dummy_map:
                target_fid, _ = dummy_map[(fid, atom.GetIdx())]
                if target_fid in frag_set:
                    is_internal_dummy = True

            if is_internal_dummy:
                continue

            new_idx = combined.AddAtom(atom)
            old_to_new_idx[(fid, atom.GetIdx())] = new_idx

    # 2) 复制每个 fragment 内部的键
    for fid in frag_ids:
        mol_f = frags_mol_list[fid]
        for bond in mol_f.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            key_u = (fid, u)
            key_v = (fid, v)
            if key_u in old_to_new_idx and key_v in old_to_new_idx:
                nu = old_to_new_idx[key_u]
                nv = old_to_new_idx[key_v]
                if combined.GetBondBetweenAtoms(nu, nv) is None:
                    combined.AddBond(nu, nv, bond.GetBondType())

    # 3) 恢复子图内部因 dummy 被跳过而缺失的键（stitch）
    processed_pairs: set[tuple[int, int]] = set()
    for fid in frag_ids:
        mol_f = frags_mol_list[fid]
        for atom in mol_f.GetAtoms():
            if atom.GetAtomicNum() != 0:
                continue
            key = (fid, atom.GetIdx())
            if key not in dummy_map:
                continue
            target_fid, target_orig_idx = dummy_map[key]
            if target_fid not in frag_set:
                continue

            # 源：本 fragment 中 dummy 的邻居真实原子
            neighbors = atom.GetNeighbors()
            if not neighbors:
                continue
            src_real = neighbors[0]
            src_new_idx = old_to_new_idx.get((fid, src_real.GetIdx()))
            if src_new_idx is None:
                continue

            # 目标：另一个 fragment 中 orig_idx 对应的真实原子
            target_mol = frags_mol_list[target_fid]
            tgt_atom_idx = -1
            for at in target_mol.GetAtoms():
                if at.HasProp("orig_idx") and at.GetIntProp("orig_idx") == target_orig_idx:
                    tgt_atom_idx = at.GetIdx()
                    break
            if tgt_atom_idx == -1:
                continue
            tgt_new_idx = old_to_new_idx.get((target_fid, tgt_atom_idx))
            if tgt_new_idx is None:
                continue

            pair = tuple(sorted((src_new_idx, tgt_new_idx)))
            if pair in processed_pairs:
                continue
            combined.AddBond(src_new_idx, tgt_new_idx, Chem.BondType.SINGLE)
            processed_pairs.add(pair)

    # 4) Sanitize 并输出 SMILES
    try:
        mol_res = combined.GetMol()
        Chem.SanitizeMol(mol_res)
        return mol_to_smi(mol_res)
    except Exception:
        return ""


def macfrag_enumerate_atom_sets(
    mol: Chem.Mol,
    max_blocks: int = 4,
) -> list[tuple[set[int], str]]:
    """
    在“无显式氢”的 RDKit Mol 上执行 MacFrag：

      - 先做 BRICS 分块（base fragments）；
      - 在 fragment graph 上枚举 size <= max_blocks 的联通子图；
      - 对每个子图用 stitching 生成一个宏碎片：
          * 返回：[(宏碎片的原子 index 集合, 宏碎片 SMILES), ...]
    """
    frags_mol_list, base_groups_vanilla = macfrag_get_brics_components(mol)

    # 如果完全没切开，退化为一个整体 fragment
    if not base_groups_vanilla:
        base_groups_vanilla = [set(range(mol.GetNumAtoms()))]
        if not frags_mol_list:
            frags_mol_list = [mol]

    adj, dummy_map = macfrag_analyze_connections(frags_mol_list, mol)
    subgraphs = macfrag_bfs_enumerate_subgraphs(adj, max_k=max_blocks)

    results: list[tuple[set[int], str]] = []
    for frag_ids in subgraphs:
        macro_smi = stitch_macro_fragment(frag_ids, frags_mol_list, dummy_map)
        if not macro_smi:
            continue

        combined_atoms: set[int] = set()
        for fid in frag_ids:
            if 0 <= fid < len(base_groups_vanilla):
                combined_atoms.update(base_groups_vanilla[fid])

        if combined_atoms:
            results.append((combined_atoms, macro_smi))

    return results

# ---------------------------------------------------------------------------
# MacFragFragmentizer：实现 BaseFragmentizer 接口（MacFrag 宏碎片）
# ---------------------------------------------------------------------------

class MacFragFragmentizer(BaseFragmentizer):
    """
    Fragmentizer implementing MacFrag (Multi-scale BRICS):

      - 先基于 BRICS 得到 base fragments；
      - 在 fragment graph 上枚举 size <= max_blocks 的联通子图；
      - 每个子图缝合成一个“宏碎片”，作为最终 fragment；

    vocab.txt 用来：
      - 将宏碎片的 SMILES 映射为 vocab id；
      - 记录频次（可选）。

    这里不再区分 vanilla / overlap 模式（那主要是 base fragment 层面的区别），
    fragment graph 里真正的 overlap 仍由 featurizer 的 k_overlap 控制。
    """

    def __init__(self, vocab_path: str, max_blocks: int | None = None):
        self.vocab_path = vocab_path
        # max_blocks 至少为 1，默认用 4（与沙盒脚本一致）
        if max_blocks is None or max_blocks <= 0:
            max_blocks = 4
        self.max_blocks = int(max_blocks)

        with open(vocab_path, "r") as f:
            lines = f.read().strip().splitlines()

        self.smi_to_id: Dict[str, int] = {}
        self.id_to_smi: List[str] = []
        self.vocab_freq: Dict[str, int] = {}

        # 第一行是 JSON header，从第二行开始读 SMILES\tatom_num\tfreq
        for line in lines[1:]:
            if not line.strip():
                continue
            smi, _, freq = line.strip().split("\t")
            if smi not in self.smi_to_id:
                self.smi_to_id[smi] = len(self.id_to_smi)
                self.id_to_smi.append(smi)
            self.vocab_freq[smi] = int(freq)

        # UNK token
        self.unk_token = "<unk>"
        if self.unk_token not in self.smi_to_id:
            self.smi_to_id[self.unk_token] = len(self.id_to_smi)
            self.id_to_smi.append(self.unk_token)
        self.unk_id = self.smi_to_id[self.unk_token]

    def get_smi_id(self, smi: str) -> int:
        return self.smi_to_id.get(smi, self.unk_id)

    def tokenize(
        self,
        mol: Chem.Mol,
    ) -> Tuple[List[Set[int]], List[int], List[List[int]]]:
        """
        对 RDKit Mol 做 MacFrag，返回：

        group_atom_indices : list[set[int]]
            每个“宏碎片”的原子 index 集合（在原始 mol 上）。
        fragment_vocab_ids : list[int]
            每个宏碎片的主 vocab id。
        final_details : list[list[int]]
            每个宏碎片的“层次 id 列表”（这里简单用 [宏碎片 id] 占位）。
        """
        macro_items = macfrag_enumerate_atom_sets(mol, max_blocks=self.max_blocks)
        if not macro_items:
            return [], [], []

        group_atom_indices: List[Set[int]] = []
        fragment_vocab_ids: List[int] = []
        final_details: List[List[int]] = []

        for atoms, smi in macro_items:
            if not atoms:
                token_id = self.unk_id
            else:
                token_id = self.get_smi_id(smi)
            group_atom_indices.append(set(atoms))
            fragment_vocab_ids.append(token_id)
            final_details.append([token_id])

        return group_atom_indices, fragment_vocab_ids, final_details




def bemis_murcko_partition_atom_sets(
    mol: Chem.Mol,
) -> Tuple[List[Set[int]], List[str]]:
    """
    Robust Bemis–Murcko 分割：

    返回：
      - groups : 每个 fragment 的原子 index 集合（在原始 mol 上）
      - types  : 对应 fragment 的类型字符串：'Ring' / 'Linker' / 'Sidechain' / 'Whole'
    """
    # 在副本上操作，避免污染原 Mol（featurizer 还会用原 Mol）
    mol_work = Chem.Mol(mol)
    for atom in mol_work.GetAtoms():
        atom.SetIntProp("orig_idx", atom.GetIdx())

    # 1) Murcko Scaffold（核心骨架）
    core = MurckoScaffold.GetScaffoldForMol(mol_work)
    core_indices: Set[int] = set()
    if core.GetNumAtoms() > 0:
        matches = mol_work.GetSubstructMatches(core, uniquify=True, maxMatches=1)       
        if matches:
            # 一般只要第一组匹配即可
            core_indices = set(matches[0])

    # 2) 标记原子角色：
    #    2 = Ring, 1 = Linker(in scaffold but not ring), 0 = Sidechain
    atom_roles: Dict[int, int] = {}
    for atom in mol_work.GetAtoms():
        idx = atom.GetIdx()
        if atom.IsInRing():
            atom_roles[idx] = 2
        elif idx in core_indices:
            atom_roles[idx] = 1
        else:
            atom_roles[idx] = 0

    # 3) 找需要切断的键
    bonds_to_cut: List[int] = []
    for bond in mol_work.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        role_u = atom_roles[u]
        role_v = atom_roles[v]

        should_cut = False

        # 不同角色之间的键一律切
        if role_u != role_v:
            should_cut = True
        # Ring-Ring 但该键不在环上（比如联苯），也切
        elif role_u == 2 and role_v == 2 and not bond.IsInRing():
            should_cut = True

        if should_cut:
            bonds_to_cut.append(bond.GetIdx())

    # 4) 如果没有切点，直接返回整个分子
    if not bonds_to_cut:
        return [set(range(mol_work.GetNumAtoms()))], ["Whole"]

    fragmented = Chem.FragmentOnBonds(
        mol_work,
        bonds_to_cut,
        dummyLabels=[(0, 0)] * len(bonds_to_cut),
    )

    # 5) 收集 fragment，并映射回原子索引 + 类型
    frag_tuples = Chem.GetMolFrags(fragmented, asMols=False)

    groups: List[Set[int]] = []
    types: List[str] = []

    for frag_indices in frag_tuples:
        cur: Set[int] = set()
        frag_role = -1

        for idx in frag_indices:
            atom = fragmented.GetAtomWithIdx(idx)
            if atom.HasProp("orig_idx"):
                orig_idx = atom.GetIntProp("orig_idx")
                cur.add(orig_idx)
                if frag_role == -1:
                    frag_role = atom_roles[orig_idx]
                else:
                    frag_role = max(frag_role, atom_roles[orig_idx])

        if cur:
            groups.append(cur)
            if frag_role == 2:
                types.append("Ring")
            elif frag_role == 1:
                types.append("Linker")
            elif frag_role == 0:
                types.append("Sidechain")
            else:
                types.append("Unknown")

    return groups, types

class BmFragmentizer(BaseFragmentizer):
    """
    Fragmentizer implementing robust Bemis–Murcko partition.

    mode:
      - "vanilla" : 严格 Bemis–Murcko 分割（fragment 间不重叠）
      - "overlap" : 用 k=1 pseudo-overlap（和 BRICS / rBRICS 一样）

    同样依赖 vocab.txt：
      - fragment SMILES -> vocab id
      - 频次信息可选
    """

    def __init__(self, vocab_path: str, mode: str = "overlap"):
        mode = mode.lower()
        if mode not in ("vanilla", "overlap"):
            raise ValueError(
                f"BmFragmentizer mode must be 'vanilla' or 'overlap', got {mode}"
            )
        self.mode = mode
        self.vocab_path = vocab_path

        with open(vocab_path, "r") as f:
            lines = f.read().strip().splitlines()

        self.smi_to_id: Dict[str, int] = {}
        self.id_to_smi: List[str] = []
        self.vocab_freq: Dict[str, int] = {}

        # 第一行是 JSON header，从第二行开始
        for line in lines[1:]:
            if not line.strip():
                continue
            smi, _, freq = line.strip().split("\t")
            if smi not in self.smi_to_id:
                self.smi_to_id[smi] = len(self.id_to_smi)
                self.id_to_smi.append(smi)
            self.vocab_freq[smi] = int(freq)

        # UNK token
        self.unk_token = "<unk>"
        if self.unk_token not in self.smi_to_id:
            self.smi_to_id[self.unk_token] = len(self.id_to_smi)
            self.id_to_smi.append(self.unk_token)
        self.unk_id = self.smi_to_id[self.unk_token]

    def get_smi_id(self, smi: str) -> int:
        return self.smi_to_id.get(smi, self.unk_id)

    def tokenize(
        self,
        mol: Chem.Mol,
    ) -> Tuple[List[Set[int]], List[int], List[List[int]]]:
        """
        对 RDKit Mol 做 robust BM partition，返回：

        group_atom_indices : list[set[int]]
            每个 fragment 的原子 index 集合（在原始 mol 上）。
        fragment_vocab_ids : list[int]
            每个 fragment 的主 vocab id。
        final_details : list[list[int]]
            每个 fragment 的“层次 id 列表”（BM 本身没有 merge history，这里用 [主 id] 占位）。
        """
        groups_vanilla, frag_types = bemis_murcko_partition_atom_sets(mol)
        if not groups_vanilla:
            return [], [], []

        # 直接重用 BRICS 的 inter_bonds + overlap 逻辑
        inter_bonds = compute_inter_bonds_from_partition(mol, groups_vanilla)

        if self.mode == "overlap":
            groups = apply_overlap_k1(groups_vanilla, inter_bonds)
        else:
            groups = groups_vanilla

        fragment_vocab_ids: List[int] = []
        final_details: List[List[int]] = []

        for atoms in groups:
            atom_list = list(atoms)
            if not atom_list:
                fragment_vocab_ids.append(self.unk_id)
                final_details.append([self.unk_id])
                continue

            submol = get_submol(mol, atom_list)
            smi = mol_to_smi(submol)
            token_id = self.get_smi_id(smi)
            fragment_vocab_ids.append(token_id)
            final_details.append([token_id])

        # 目前 CGMNet 不用 frag_types，这里先不返回；以后需要可以再挂到 graph.ndata 里
        return groups, fragment_vocab_ids, final_details


# ---------------------------------------------------------------------------
# Ertl Functional Groups + Skeleton helpers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# EFG helpers: Ertl Functional Groups + skeleton partition
# ---------------------------------------------------------------------------

def ertl_fg_partition_atom_sets(mol: Chem.Mol) -> List[Set[int]]:
    """
    用 efgs.get_fgs(mol) 做 Ertl FG 分割，然后把剩余原子按连通分量分成 skeleton 片段。

    返回：
      groups : list[set[int]]
        每个元素是一组原子索引（在 mol 上的 index），
        既包括 FunctionalGroup，也包括 Skeleton 片段。
    """
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return []

    try:
        # efgs.get_fgs(mol) 预期返回可迭代，每个元素是一组 atom indices
        fgs_raw = efgs.get_fgs(mol)
    except Exception as e:
        # 兜底：EFG 失败时当成一个整体 fragment
        # （和沙盒脚本里 Warning + Whole 一致）
        # print(f"[EFG] efgs.get_fgs failed: {e}")
        return [set(range(num_atoms))]

    groups: List[Set[int]] = []
    covered_atoms: Set[int] = set()

    # 1. Functional Groups
    for fg in fgs_raw:
        fg_set = set(int(a) for a in fg)
        if not fg_set:
            continue
        groups.append(fg_set)
        covered_atoms.update(fg_set)

    # 2. Skeleton atoms = 还没有被 FG 覆盖的原子
    all_atoms = set(range(num_atoms))
    skeleton_atoms = all_atoms - covered_atoms
    if not skeleton_atoms:
        # 整个分子都被 FG 覆盖了
        return groups

    # 3. 把 skeleton_atoms 按 connectivity 划分成若干 skeleton 片段
    visited: Set[int] = set()
    for start in sorted(skeleton_atoms):
        if start in visited:
            continue
        comp: Set[int] = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.add(cur)
            atom = mol.GetAtomWithIdx(cur)
            for nb in atom.GetNeighbors():
                nid = nb.GetIdx()
                if nid in skeleton_atoms and nid not in visited:
                    stack.append(nid)
        if comp:
            groups.append(comp)

    if not groups:
        return [set(range(num_atoms))]
    return groups

# ---------------------------------------------------------------------------
# EfgFragmentizer：实现 BaseFragmentizer 接口（Ertl FGs + skeleton）
# ---------------------------------------------------------------------------

class EfgFragmentizer(BaseFragmentizer):
    """
    Fragmentizer implementing Ertl functional groups + skeleton partition.

    mode:
      - "vanilla" : 片段之间不重叠；
      - "overlap" : 在每条跨 fragment 键处引入 overlap=1，
                    以便 fragment graph 能按重叠构边（和 BRICS 一致）。

    仍然依赖 vocab.txt：
      - fragment SMILES -> vocab id
      - 频次作为可选信息（词表里第三列）
    """

    def __init__(self, vocab_path: str, mode: str = "overlap"):
        mode = mode.lower()
        if mode not in ("vanilla", "overlap"):
            raise ValueError(f"EfgFragmentizer mode must be 'vanilla' or 'overlap', got {mode}")
        self.mode = mode
        self.vocab_path = vocab_path

        with open(vocab_path, "r") as f:
            lines = f.read().strip().splitlines()

        self.smi_to_id: Dict[str, int] = {}
        self.id_to_smi: List[str] = []
        self.vocab_freq: Dict[str, int] = {}

        # 第一行是 JSON header，从第二行开始
        for line in lines[1:]:
            if not line.strip():
                continue
            smi, _, freq = line.strip().split("\t")
            if smi not in self.smi_to_id:
                self.smi_to_id[smi] = len(self.id_to_smi)
                self.id_to_smi.append(smi)
            self.vocab_freq[smi] = int(freq)

        # UNK token
        self.unk_token = "<unk>"
        if self.unk_token not in self.smi_to_id:
            self.smi_to_id[self.unk_token] = len(self.id_to_smi)
            self.id_to_smi.append(self.unk_token)
        self.unk_id = self.smi_to_id[self.unk_token]

    def get_smi_id(self, smi: str) -> int:
        return self.smi_to_id.get(smi, self.unk_id)

    def tokenize(
        self,
        mol: Chem.Mol,
    ) -> Tuple[List[Set[int]], List[int], List[List[int]]]:
        """
        对 RDKit Mol 做 Ertl FG + skeleton partition，返回：

        group_atom_indices : list[set[int]]
            每个 fragment 的原子 index 集合（在原始 mol 上）。
        fragment_vocab_ids : list[int]
            每个 fragment 的主 vocab id。
        final_details : list[list[int]]
            每个 fragment 的“层次 id 列表”
            （EFG 没有合并历史，这里用 [主 id] 占位）。
        """
        groups_vanilla = ertl_fg_partition_atom_sets(mol)
        if not groups_vanilla:
            return [], [], []

        # 重用 BRICS 的 inter_bonds 计算 + overlap 逻辑
        inter_bonds = compute_inter_bonds_from_partition(mol, groups_vanilla)

        if self.mode == "overlap":
            groups = apply_overlap_k1(groups_vanilla, inter_bonds)
        else:
            groups = groups_vanilla

        fragment_vocab_ids: List[int] = []
        final_details: List[List[int]] = []

        for atoms in groups:
            atom_list = list(atoms)
            if not atom_list:
                fragment_vocab_ids.append(self.unk_id)
                final_details.append([self.unk_id])
                continue

            submol = get_submol(mol, atom_list)
            smi = mol_to_smi(submol)
            token_id = self.get_smi_id(smi)
            fragment_vocab_ids.append(token_id)
            final_details.append([token_id])

        return groups, fragment_vocab_ids, final_details


# ---------------------------------------------------------------------------
# AccFG helpers: 功能团 + Skeleton 分割
# ---------------------------------------------------------------------------

def accfg_partition_atom_sets(
    mol: Chem.Mol,
) -> Tuple[List[Set[int]], List[str]]:
    """
    使用 AccFG 对 RDKit Mol 分割：
      - 返回 (groups, types)
        groups: 每个 fragment 的 atom index 集合（无显式 H 的 mol 上）
        types : 每个 fragment 的类型名字（功能团名 / 'Skeleton'）

    如果 AccFG 失败，就退化为整个分子一个 fragment。
    """
    if ACCFG_ENGINE is None:
        # 没有正确初始化 AccFG
        num_atoms = mol.GetNumAtoms()
        return [set(range(num_atoms))], ["Whole"]

    groups: List[Set[int]] = []
    types: List[str] = []
    covered_atoms: Set[int] = set()

    try:
        # AccFG.run_mol(mol, show_atoms=True) -> {'FG name': [[atom_idx...], ...], ...}
        fgs_dict = ACCFG_ENGINE.run_mol(mol, show_atoms=True, show_graph=False)
    except Exception as e:
        print(f"[AccFG] run_mol failed: {e}")
        num_atoms = mol.GetNumAtoms()
        return [set(range(num_atoms))], ["Whole"]

    # 拉平成 (名称, 原子集合) 列表
    all_found_fgs: List[Tuple[str, Set[int]]] = []
    for name, atom_lists in fgs_dict.items():
        for atom_list in atom_lists:
            atom_set = set(atom_list)
            if atom_set:
                all_found_fgs.append((name, atom_set))

    # 大 fragment 优先（比如“氨基酸”优先于“胺”）
    all_found_fgs.sort(key=lambda x: len(x[1]), reverse=True)

    for name, atom_set in all_found_fgs:
        # 已经完全被覆盖了就跳过
        if atom_set.isdisjoint(covered_atoms):
            groups.append(atom_set)
            types.append(name)
            covered_atoms.update(atom_set)
        else:
            # 有重叠：按沙盒思路，不拆功能团，剩余部分交给 Skeleton
            remaining = atom_set - covered_atoms
            if remaining:
                # 为保证语义完整，不把残余当作“同一个 FG”
                # 它们会在 skeleton 分组里被吃掉
                pass

    # Skeleton = 所有没被任何 FG 覆盖的原子
    all_atoms = set(range(mol.GetNumAtoms()))
    skeleton_atoms = all_atoms - covered_atoms

    if skeleton_atoms:
        visited: Set[int] = set()
        for idx in sorted(skeleton_atoms):
            if idx in visited:
                continue
            comp: Set[int] = set()
            stack = [idx]
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                comp.add(cur)
                atom = mol.GetAtomWithIdx(cur)
                for nb in atom.GetNeighbors():
                    j = nb.GetIdx()
                    if j in skeleton_atoms and j not in visited:
                        stack.append(j)
            if comp:
                groups.append(comp)
                types.append("Skeleton")

    if not groups:
        # 极端兜底：整个分子一个 fragment
        return [set(range(mol.GetNumAtoms()))], ["Whole"]

    return groups, types


class AccfgFragmentizer(BaseFragmentizer):
    """
    Fragmentizer implementing AccFG functional groups + skeleton 分割。

    mode:
      - "vanilla" : 原始 AccFG+Skeleton partition（fragment 间不重叠）
      - "overlap" : 在每条跨 fragment 键处引入 overlap=1（适配 CGMNet fragment graph）
    """

    def __init__(self, vocab_path: str, mode: str = "overlap"):
        mode = mode.lower()
        if mode not in ("vanilla", "overlap"):
            raise ValueError(f"AccfgFragmentizer mode must be 'vanilla' or 'overlap', got {mode}")
        self.mode = mode
        self.vocab_path = vocab_path

        # 读 vocab（格式与其它 fragmentizer 一致）
        with open(vocab_path, "r") as f:
            lines = f.read().strip().splitlines()

        self.smi_to_id: Dict[str, int] = {}
        self.id_to_smi: List[str] = []
        self.vocab_freq: Dict[str, int] = {}

        for line in lines[1:]:  # 第一行是 JSON header
            if not line.strip():
                continue
            smi, _, freq = line.strip().split("\t")
            if smi not in self.smi_to_id:
                self.smi_to_id[smi] = len(self.id_to_smi)
                self.id_to_smi.append(smi)
            self.vocab_freq[smi] = int(freq)

        # UNK token
        self.unk_token = "<unk>"
        if self.unk_token not in self.smi_to_id:
            self.smi_to_id[self.unk_token] = len(self.id_to_smi)
            self.id_to_smi.append(self.unk_token)
        self.unk_id = self.smi_to_id[self.unk_token]

    def get_smi_id(self, smi: str) -> int:
        return self.smi_to_id.get(smi, self.unk_id)

    def tokenize(
        self,
        mol: Chem.Mol,
    ) -> Tuple[List[Set[int]], List[int], List[List[int]]]:
        """
        对 RDKit Mol 做 AccFG+Skeleton 分割。

        Returns
        -------
        group_atom_indices : list[set[int]]
        fragment_vocab_ids : list[int]
        final_details      : list[list[int]]  # AccFG 没有 merge history，这里用 [主 id] 占位
        """
        groups_vanilla, frag_types = accfg_partition_atom_sets(mol)
        if not groups_vanilla:
            return [], [], []

        # 复用 BRICS 的 “跨 fragment 键” + overlap=k1 逻辑
        inter_bonds = compute_inter_bonds_from_partition(mol, groups_vanilla)

        if self.mode == "overlap":
            groups = apply_overlap_k1(groups_vanilla, inter_bonds)
        else:
            groups = groups_vanilla

        fragment_vocab_ids: List[int] = []
        final_details: List[List[int]] = []

        for atoms in groups:
            atom_list = list(atoms)
            if not atom_list:
                fragment_vocab_ids.append(self.unk_id)
                final_details.append([self.unk_id])
                continue

            submol = get_submol(mol, atom_list)
            smi = mol_to_smi(submol)
            token_id = self.get_smi_id(smi)
            fragment_vocab_ids.append(token_id)
            final_details.append([token_id])

        return groups, fragment_vocab_ids, final_details


# ---------------------------------------------------------------------------
# Louvain helpers: L^k(G) 上的社区检测 → atom index fragment sets
# ---------------------------------------------------------------------------

def _flatten_louvain_node(node):
    """递归展开 line-graph 节点（tuple 嵌套）到原始原子索引."""
    if isinstance(node, int):
        yield node
    elif hasattr(node, "__iter__"):
        for item in node:
            yield from _flatten_louvain_node(item)


def louvain_partition_atom_sets(
    mol: Chem.Mol,
    k_line: int = 0,
) -> List[Set[int]]:
    """
    纯 Louvain 碎片化：

      - k_line = 0：直接在原子图 G 上做 Louvain，得到不重叠的 fragment；
      - k_line >= 1：对 G 连续做 k_line 次 line graph， 在 L^k(G) 上做 Louvain，
                     然后把社区里的“节点”（边/边的边/……）flatten 回原子 index，
                     得到可以重叠的 fragment。

    返回：每个 fragment 在原始 mol 上的 atom index 集合。
    """
    if community_louvain is None:
        raise ImportError(
            "python-louvain (import community as community_louvain) is required "
            "for 'louvain' fragmentation. Please install it via: pip install python-louvain"
        )

    # 1. 原子图 G_0
    g0 = nx.Graph()
    for atom in mol.GetAtoms():
        g0.add_node(atom.GetIdx())
    for bond in mol.GetBonds():
        g0.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    if g0.number_of_nodes() == 0:
        return []

    target_graph = g0

    # 2. 做 k_line 次 line graph: G -> L(G) -> L^2(G) ...
    if k_line > 0:
        current_g = g0
        for _ in range(k_line):
            if current_g.number_of_edges() == 0:
                break
            current_g = nx.line_graph(current_g)
        target_graph = current_g

    # 退化情况：L^k(G) 没节点，那就仍然视作整体一个 fragment
    if target_graph.number_of_nodes() == 0:
        return [set(g0.nodes())]

    # 3. Louvain 社区检测
    partition = community_louvain.best_partition(target_graph, random_state=42)
    communities: Dict[int, List] = {}
    for node, cid in partition.items():
        communities.setdefault(cid, []).append(node)

    # 4. 展开社区 → atom index sets
    fragments: List[Set[int]] = []
    for cid in sorted(communities.keys()):
        atom_indices: Set[int] = set()
        for node in communities[cid]:
            for atom_idx in _flatten_louvain_node(node):
                atom_indices.add(int(atom_idx))
        if atom_indices:
            fragments.append(atom_indices)

    if not fragments:
        # 兜底：至少给一个“全分子” fragment
        return [set(g0.nodes())]

    return fragments


# ---------------------------------------------------------------------------
# LouvainFragmentizer：实现 BaseFragmentizer 接口（graph-based）
# ---------------------------------------------------------------------------

class LouvainFragmentizer(BaseFragmentizer):
    """
    Fragmentizer implementing pure Louvain community detection on molecular graphs.

    参数：
      - k_line: 在 L^k(G) 上做 Louvain（k_line = 0/1/2/...）。
        这里我们把它和 vocab 里的 line_order / CLI 的 --order 对齐。
    """

    def __init__(self, vocab_path: str, k_line: int = 0):
        if community_louvain is None:
            raise ImportError(
                "python-louvain is required for LouvainFragmentizer. "
                "Please install it via: pip install python-louvain"
            )

        self.vocab_path = vocab_path
        self.k_line = int(k_line)

        with open(vocab_path, "r") as f:
            lines = f.read().strip().splitlines()

        self.smi_to_id: Dict[str, int] = {}
        self.id_to_smi: List[str] = []
        self.vocab_freq: Dict[str, int] = {}

        # 第一行 JSON 是 meta，从第二行开始 vocab 列表
        for line in lines[1:]:
            if not line.strip():
                continue
            smi, _, freq = line.strip().split("\t")
            if smi not in self.smi_to_id:
                self.smi_to_id[smi] = len(self.id_to_smi)
                self.id_to_smi.append(smi)
            self.vocab_freq[smi] = int(freq)

        # UNK token
        self.unk_token = "<unk>"
        if self.unk_token not in self.smi_to_id:
            self.smi_to_id[self.unk_token] = len(self.id_to_smi)
            self.id_to_smi.append(self.unk_token)
        self.unk_id = self.smi_to_id[self.unk_token]

    def get_smi_id(self, smi: str) -> int:
        return self.smi_to_id.get(smi, self.unk_id)

    def tokenize(
        self,
        mol: Chem.Mol,
    ) -> Tuple[List[Set[int]], List[int], List[List[int]]]:
        """
        对 RDKit Mol 做 Louvain-based 分割：

        返回：
          - group_atom_indices : list[set[int]]
          - fragment_vocab_ids : list[int]
          - final_details      : list[list[int]]（Louvain 没 merge 历史，用 [主 id] 占位）
        """
        groups = louvain_partition_atom_sets(mol, k_line=self.k_line)
        if not groups:
            return [], [], []

        fragment_vocab_ids: List[int] = []
        final_details: List[List[int]] = []

        for atoms in groups:
            atom_list = list(atoms)
            if not atom_list:
                fragment_vocab_ids.append(self.unk_id)
                final_details.append([self.unk_id])
                continue

            submol = get_submol(mol, atom_list)
            smi = mol_to_smi(submol)
            token_id = self.get_smi_id(smi)
            fragment_vocab_ids.append(token_id)
            final_details.append([token_id])

        return groups, fragment_vocab_ids, final_details


# ---------------------------------------------------------------------------
# 预留：以后可以在这里增加 EFG / AccFG 等算法
# ---------------------------------------------------------------------------

# class EFGFragmentizer(BaseFragmentizer):
#     ...
#
# class AccFGFragmentizer(BaseFragmentizer):
#     ...

