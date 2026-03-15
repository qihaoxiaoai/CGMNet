# scripts/02_generate_features.py
"""
Generate knowledge features for datasets.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from cgmnet.utils.fingerprint import get_batch_fingerprints, FINGERPRINT_DIMENSIONS


def process_smiles_list(smiles_list: list, output_dir: Path, features: list, n_jobs: int):
    """Generate and save features for a list of SMILES."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Outputs will be saved to: {output_dir} ---")
    for feature_name in tqdm(features, desc="Generating features"):
        feature_array = get_batch_fingerprints(
            smiles_list,
            name=feature_name,
            n_jobs=n_jobs
        )

        output_path = output_dir / f"{feature_name}.npy"
        np.save(output_path, feature_array)
        print(f"Saved {feature_array.shape} array to {output_path}")


def main():
    """Parse arguments and run feature generation."""
    parser = argparse.ArgumentParser(
        description="Generate knowledge features with flexible output directory logic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=['pretrain', 'finetune'],
        help="The mode of operation: 'pretrain' or 'finetune'."
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="Input path: .smi file for pretrain, data root for finetune."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Optional: Base directory to save 'features'.\n"
            "  - pretrain: features will be saved to <output_dir>/features.\n"
            "  - finetune: features will be saved to <output_dir>/<dataset_name>/features.\n"
            "If not given:\n"
            "  - pretrain: defaults to input_path.parent.parent / 'features'.\n"
            "  - finetune: defaults to <dataset_root>/<dataset_name>/features."
        )
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        help="[Finetune mode only] A list of dataset names to process (e.g., bace clintox)."
    )
    parser.add_argument(
        "--features",
        type=str,
        nargs='+',
        default=['ecfp', 'maccs', 'torsion', 'md'],
        choices=list(FINGERPRINT_DIMENSIONS.keys()),
        help="List of features/fingerprints to generate."
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=16,
        help="Number of parallel CPU cores to use."
    )
    args = parser.parse_args()

    if args.mode == 'pretrain':
        print(f"--- Running in Pre-training Mode for: {args.input_path.name} ---")
        if not args.input_path.is_file():
            raise FileNotFoundError(
                f"Input for pretrain mode must be a file: {args.input_path}"
            )

        if args.output_dir:
            output_base_dir = args.output_dir
        else:
            output_base_dir = args.input_path.resolve().parent.parent

        features_dir = output_base_dir / "features"

        smiles_list = pd.read_csv(
            args.input_path,
            header=None,
            names=['smiles']
        )['smiles'].dropna().tolist()

        process_smiles_list(smiles_list, features_dir, args.features, args.n_jobs)

    elif args.mode == 'finetune':
        print(f"--- Running in Fine-tuning Mode for data root: {args.input_path} ---")
        if not args.datasets:
            parser.error("--datasets must be provided in 'finetune' mode.")

        for name in args.datasets:
            print(f"\n--- Processing dataset: {name} ---")
            csv_path = args.input_path / name / f"{name}.csv"
            if not csv_path.exists():
                print(f"Warning: CSV file for '{name}' not found. Skipping.")
                continue

            if args.output_dir:
                output_base_dir = args.output_dir / name
            else:
                output_base_dir = csv_path.parent

            features_dir = output_base_dir / "features"
            smiles_list = pd.read_csv(csv_path)['smiles'].dropna().tolist()
            process_smiles_list(smiles_list, features_dir, args.features, args.n_jobs)

    print("\n✅ Feature generation complete!")


if __name__ == "__main__":
    main()
