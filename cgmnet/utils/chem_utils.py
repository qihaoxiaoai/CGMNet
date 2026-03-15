# cgmnet/utils/chem_utils_optimized.py
"""
A central toolkit for cheminformatics operations using RDKit.
This is the OPTIMIZED version with a robust atom counter.
"""
from functools import partial
from typing import List, Set
import re

from dgllife.utils.featurizers import (
    ConcatFeaturizer,
    atom_chirality_type_one_hot,
    atom_degree_one_hot,
    atom_formal_charge,
    atom_hybridization_one_hot,
    atom_is_aromatic,
    atom_is_in_ring,
    atom_mass,
    atomic_number_one_hot,
    atom_num_radical_electrons_one_hot,
    atom_total_num_H_one_hot,
    bond_is_conjugated,
    bond_is_in_ring,
    bond_stereo_one_hot,
    bond_type_one_hot,
)
from rdkit import Chem

# --- Constants (Unchanged) ---
ATOM_FEATURE_DIM = 137
BOND_FEATURE_DIM = 14

# --- A definitive set of all chemical element symbols for validation (Unchanged) ---
PERIODIC_TABLE_ELEMENTS = {
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si',
    'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
    'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
    'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho',
    'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np',
    'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg',
    'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
    'b', 'c', 'n', 'o', 'p', 's', 'te', 'se', 'ge', 'as', '*'
}

# ============================ FINAL ROBUST FIX START ============================
# The cnt_atom function is enhanced based on the original structure.
# It now correctly extracts and validates element symbols from all SMILES/SMARTS constructs.

def cnt_atom(smi: str) -> int:
    """
    Counts the number of heavy atoms in a SMILES/SMARTS string robustly and comprehensively.
    It uses a regex to tokenize the string into atoms, then validates each token.
    This version enhances the validation logic to be fully comprehensive.
    """
    if not isinstance(smi, str):
        return 0
    
    # This regex for tokenizing the string into "atom blocks" is kept as it is robust and correct.
    atom_finder = re.compile(r'(\[[^\]]+\]|Br|Cl|[A-Za-z]|\*)')
    matches = atom_finder.findall(smi)
    
    heavy_atom_count = 0
    for match in matches:
        element_symbol = ''
        if match.startswith('['):
            # ENHANCEMENT: More precise extraction of the element symbol from within the brackets.
            # This regex finds the first one or two alphabetic characters, which is the standard
            # representation for an element symbol (e.g., C, Cl, Se, Au).
            symbol_match = re.search(r'[A-Za-z][a-z]?', match)
            if symbol_match:
                element_symbol = symbol_match.group(0)
            elif '*' in match:
                element_symbol = '*' # Handle wildcard case like [*]
        else:
            # For non-bracketed atoms, the match itself is the symbol.
            element_symbol = match

        # Final validation step (logic remains the same, but now receives a more accurate symbol).
        # We count it if it's a valid symbol and not Hydrogen.
        if element_symbol and element_symbol != 'H' and element_symbol in PERIODIC_TABLE_ELEMENTS:
            heavy_atom_count += 1
            
    return heavy_atom_count
# ============================= FINAL ROBUST FIX END =============================


# --- Original Utility Functions (Unchanged) ---

def smi_to_mol(smiles: str, kekulize: bool = False, sanitize: bool = True) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol and kekulize:
        Chem.Kekulize(mol, True)
    return mol

def mol_to_smi(mol: Chem.Mol, canonical: bool = True) -> str:
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=canonical)

def get_submol(mol: Chem.Mol, atom_indices: List[int]) -> Chem.Mol:
    if not atom_indices:
        return None
    if len(atom_indices) == 1:
        atom = mol.GetAtomWithIdx(atom_indices[0])
        return Chem.MolFromSmarts(f"[{atom.GetSmarts()}]")

    bond_indices = []
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in atom_indices and bond.GetEndAtomIdx() in atom_indices:
            bond_indices.append(bond.GetIdx())
    
    if not bond_indices and len(atom_indices) > 1:
        smarts_parts = [mol.GetAtomWithIdx(aid).GetSmarts() for aid in atom_indices]
        return Chem.MolFromSmarts(".".join(smarts_parts))

    return Chem.PathToSubmol(mol, bond_indices)

# --- Featurizer definitions (unchanged) ---
atom_featurizer_functions = [
    partial(atomic_number_one_hot, allowable_set=list(range(1, 101)), encode_unknown=True),
    partial(atom_degree_one_hot, allowable_set=list(range(0, 11)), encode_unknown=True),
    atom_formal_charge,
    partial(atom_num_radical_electrons_one_hot, allowable_set=list(range(0, 5)), encode_unknown=True),
    partial(atom_hybridization_one_hot, encode_unknown=True),
    atom_is_aromatic,
    partial(atom_total_num_H_one_hot, allowable_set=list(range(0, 5)), encode_unknown=True),
    atom_is_in_ring,
    atom_chirality_type_one_hot,
    atom_mass,
]
atom_featurizer = ConcatFeaturizer(atom_featurizer_functions)

bond_featurizer = ConcatFeaturizer([
    partial(bond_type_one_hot, encode_unknown=True),
    bond_is_conjugated,
    bond_is_in_ring,
    partial(bond_stereo_one_hot, allowable_set=[
        Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS, Chem.rdchem.BondStereo.STEREOTRANS
    ], encode_unknown=True)
])
