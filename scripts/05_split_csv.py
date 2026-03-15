# scripts/create_scaffold_splits.py
"""
Create scaffold splits for a chemical dataset from a CSV file.
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm


def generate_scaffold(mol: Chem.Mol, isomeric_smiles: bool = True) -> str:
    """
    Generate a Murcko scaffold for an RDKit molecule.
    """
    try:
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)

        if scaffold_mol and scaffold_mol.GetNumAtoms() > 0:
            return Chem.MolToSmiles(scaffold_mol, isomericSmiles=isomeric_smiles)
        else:
            return Chem.MolToSmiles(mol, isomericSmiles=isomeric_smiles)
    except Exception:
        return Chem.MolToSmiles(mol, isomericSmiles=isomeric_smiles)


def create_scaffold_split(
    data_path: Path,
    smiles_col: str = "smiles",
    seed: int = 42,
    split_ratios: tuple = (0.8, 0.1, 0.1),
):
    """
    Create and save a scaffold split for a given seed.
    """
    random.seed(seed)
    np.random.seed(seed)

    df = pd.read_csv(data_path)
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in {data_path}")

    print(f"\n--- Processing for seed {seed} ---")
    scaffolds = defaultdict(list)
    for i, smi in enumerate(tqdm(df[smiles_col], desc=f"Generating scaffolds (seed={seed})")):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                scaffold = generate_scaffold(mol)
                scaffolds[scaffold].append(i)
        except Exception as e:
            print(f"Warning: Skipping invalid SMILES at index {i}: {smi} | Error: {e}")
            continue

    scaffold_sets = sorted(list(scaffolds.keys()))
    random.shuffle(scaffold_sets)

    train_cutoff = int(split_ratios[0] * len(scaffold_sets))
    valid_cutoff = int((split_ratios[0] + split_ratios[1]) * len(scaffold_sets))

    train_scaffolds = scaffold_sets[:train_cutoff]
    valid_scaffolds = scaffold_sets[train_cutoff:valid_cutoff]
    test_scaffolds = scaffold_sets[valid_cutoff:]

    train_indices = np.array([idx for s in train_scaffolds for idx in scaffolds[s]], dtype=int)
    valid_indices = np.array([idx for s in valid_scaffolds for idx in scaffolds[s]], dtype=int)
    test_indices = np.array([idx for s in test_scaffolds for idx in scaffolds[s]], dtype=int)

    print(f"Split sizes: Train={len(train_indices)}, Valid={len(valid_indices)}, Test={len(test_indices)}")

    output_dir = data_path.parent / "splits"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"scaffold-{seed}.npy"

    np.save(output_path, np.array([train_indices, valid_indices, test_indices], dtype=object))
    print(f"✅ Split file saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create scaffold splits from a CSV file.")
    parser.add_argument("--data_path", type=Path, required=True, help="Path to the input CSV data file.")
    parser.add_argument("--smiles_col", type=str, default="smiles", help="Name of the column containing SMILES strings.")
    parser.add_argument("--split_ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1], help="Train, validation, and test split ratios.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="A list of random seeds to generate splits for.")
    args = parser.parse_args()

    if not np.isclose(sum(args.split_ratios), 1.0):
        raise ValueError("Split ratios must sum to 1.")

    for seed in args.seeds:
        create_scaffold_split(args.data_path, args.smiles_col, seed, tuple(args.split_ratios))


if __name__ == "__main__":
    main()
