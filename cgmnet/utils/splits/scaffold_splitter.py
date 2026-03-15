# cgmnet/utils/splits/scaffold_splitter.py
"""
Provides functionality for splitting a chemical dataset based on molecular scaffolds.

This method helps to create more challenging and realistic train/validation/test
splits by ensuring that molecules with the same core structure (scaffold) are
kept within the same set, preventing the model from simply memorizing scaffolds.
"""
from collections import defaultdict
from typing import List, Tuple

import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold


def generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """
    Generates the Bemis-Murcko scaffold for a given SMILES string.

    Args:
        smiles: The SMILES string of the molecule.
        include_chirality: Whether to include chirality in the scaffold definition.

    Returns:
        The SMILES string of the generated scaffold.
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality
    )
    return scaffold


def generate_scaffold_split(
    smiles_list: List[str],
    fractions: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Splits a list of SMILES into train, validation, and test sets based on their
    molecular scaffolds.

    Args:
        smiles_list: A list of SMILES strings to be split.
        fractions: A tuple specifying the fractions for train, valid, and test sets.
        seed: The random seed for shuffling scaffolds.

    Returns:
        A tuple containing three lists of indices: (train_idx, valid_idx, test_idx).
    """
    assert sum(fractions) == 1.0, "Fractions must sum to 1."

    # Group molecules by their scaffold
    all_scaffolds = defaultdict(list)
    for i, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        all_scaffolds[scaffold].append(i)

    # Sort scaffold groups by size (largest first) to ensure balanced splits
    all_scaffold_sets = sorted(all_scaffolds.values(), key=len, reverse=True)

    # Shuffle the scaffold groups for randomness
    rng = np.random.default_rng(seed)
    rng.shuffle(all_scaffold_sets)

    # Greedily assign scaffolds to splits
    train_cutoff = fractions[0] * len(smiles_list)
    valid_cutoff = (fractions[0] + fractions[1]) * len(smiles_list)
    
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) < train_cutoff:
            train_idx.extend(scaffold_set)
        elif len(train_idx) + len(valid_idx) < valid_cutoff:
            valid_idx.extend(scaffold_set)
        else:
            test_idx.extend(scaffold_set)

    # Assert that splits are disjoint
    assert len(set(train_idx) & set(valid_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(valid_idx) & set(test_idx)) == 0

    return train_idx, valid_idx, test_idx
