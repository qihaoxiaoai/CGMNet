# scripts/01_build_vocabulary.py
"""
Build a fragment vocabulary for CGMNet.
"""

import argparse
import sys
from pathlib import Path
from typing import List

from tqdm import tqdm
from rdkit import Chem

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from cgmnet.data.vocabulary import build_vocabulary


def filter_smiles_file(input_path: Path, output_path: Path) -> List[str]:
    """
    Reads a SMILES file, validates each SMILES, and writes the valid ones
    to a new file. Returns the list of valid SMILES.
    """
    print(f"--- Step 1: Pre-filtering SMILES from {input_path} ---")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r") as f:
        all_smiles = [line.strip() for line in f if line.strip()]
    print(f"Found {len(all_smiles)} total SMILES. Validating...")

    valid_smiles = [
        smi
        for smi in tqdm(all_smiles, desc="Validating SMILES")
        if Chem.MolFromSmiles(smi) is not None
    ]

    print(f"Validation complete. Found {len(valid_smiles)} valid SMILES.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(valid_smiles) + "\n")
    print(f"Saved {len(valid_smiles)} valid SMILES to {output_path}")

    return valid_smiles


def main():
    parser = argparse.ArgumentParser(
        description="Build a fragment vocabulary for CGMNet (DOVE / BRICS, etc.).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_smi_path",
        type=Path,
        required=True,
        help="Path to the input file containing raw SMILES strings.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Directory to save outputs. "
            "If not provided, defaults to the input SMILES's directory."
        ),
    )
    parser.add_argument(
        "--fragments_to_mine",
        type=int,
        default=500,
        help="The number of NEW fragments to mine on top of the initial ones (only used by DOVE).",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=1,
        help=(
            "k_line: order of the line graph transformation used during vocab mining (DOVE). "
            "Default is 1 (starts with bonds)."
        ),
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=8,
        help="Number of parallel CPU cores to use (only used by DOVE).",
    )
    parser.add_argument(
        "--kekulize",
        action="store_true",
        help="Kekulize molecules before processing.",
    )
    parser.add_argument(
        "--frag_method",
        type=str,
        default="dove",
        choices=[
            "dove",
            "brics", "brics_vanilla", "brics_overlap",
            "rbrics", "rbrics_vanilla", "rbrics_overlap",
            "recap", "recap_vanilla", "recap_overlap",
            "relmole", "relmole_vanilla", "relmole_overlap",
            "accfg", "accfg_vanilla", "accfg_overlap",
            "jt", "jt_vae",
            "macfrag", "macfrag_vanilla", "macfrag_overlap",
            "bm", "bm_vanilla", "bm_overlap",
            "efgs", "efgs_vanilla", "efgs_overlap",
            "louvain",
        ],
        help=(
            "Fragmentation method used to build the vocabulary. "
            "'dove' uses the DOVE mining algorithm; "
            "'brics'/'brics_overlap' use BRICS with overlap=1; "
            "'brics_vanilla' uses vanilla BRICS partition."
            "'rbrics'/'rbrics_overlap' use ring-breaking r-BRICS with overlap=1; "
            "'rbrics_vanilla' uses vanilla r-BRICS partition. "
            "for non-dove fragments_to_mine/order/n_jobs are ignored."
            "For BRICS/r-BRICS methods, fragments_to_mine/order/n_jobs are ignored."
        ),
    )

    args = parser.parse_args()

    if args.output_dir:
        output_base_dir = args.output_dir
    else:
        output_base_dir = args.input_smi_path.parent

    cleaned_smi_path = output_base_dir / "data" / "cleaned.smi"
    vocab_path = output_base_dir / "vocabs" / "vocab.txt"

    cleaned_smi_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)

    valid_smiles = filter_smiles_file(args.input_smi_path, cleaned_smi_path)

    print(
        f"\n--- Step 2: Building vocabulary with frag_method = '{args.frag_method}' ---"
    )

    build_vocabulary(
        smiles_list=valid_smiles,
        fragments_to_mine=args.fragments_to_mine,
        output_path=str(vocab_path),
        frag_method=args.frag_method,
        order=args.order,
        n_jobs=args.n_jobs,
        kekulize=args.kekulize,
    )

    print("\nFinal vocabulary generation finished successfully!")
    print(f"Final vocabulary file saved to: {vocab_path}")


if __name__ == "__main__":
    main()
