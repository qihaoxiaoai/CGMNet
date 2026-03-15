# scripts/03_build_lmdb.py
"""
Build an LMDB database for efficient pre-training.
"""

import argparse
import io
import os
import lmdb
import sys
import torch
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import traceback

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from cgmnet.data.featurizer import CGMNetFeaturizer
from cgmnet.data.fragmentizer import load_vocab_meta

_GLOBAL_FEATURIZER = None
_GLOBAL_FRAG_METHOD = None


def _init_worker(
    vocab_path: str,
    order: int,
    max_path_len: int,
    frag_method: str,
    overlap_degree: int | None,
):
    """
    Pool initializer that constructs one featurizer per worker process.
    """
    global _GLOBAL_FEATURIZER, _GLOBAL_FRAG_METHOD

    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    _GLOBAL_FRAG_METHOD = frag_method

    _GLOBAL_FEATURIZER = CGMNetFeaturizer(
        vocab_path=vocab_path,
        order=order,
        max_path_length=max_path_len,
        frag_method=frag_method,
        overlap_degree=overlap_degree,
    )
    print(
        f"[Worker init] PID={os.getpid()} "
        f"initialized featurizer with vocab={vocab_path}, "
        f"order={order}, overlap_degree={overlap_degree}, "
        f"path_max_length={max_path_len}, frag_method={frag_method}",
        file=sys.stderr,
    )


def _worker_featurize_and_serialize(smi: str):
    """Worker function for multiprocessing. Processes a single SMILES string."""
    global _GLOBAL_FEATURIZER, _GLOBAL_FRAG_METHOD
    try:
        if _GLOBAL_FEATURIZER is None:
            raise RuntimeError("GLOBAL featurizer is not initialized in worker.")

        graph_dict = _GLOBAL_FEATURIZER(smi)
        if graph_dict is None:
            return smi, None

        buffer = io.BytesIO()
        torch.save(graph_dict, buffer)
        graph_bytes = buffer.getvalue()

        return smi, graph_bytes
    except Exception:
        print(f"ERROR processing SMILES: {smi}", file=sys.stderr)
        traceback.print_exc()
        return smi, None


def main():
    parser = argparse.ArgumentParser(
        description="Build an LMDB database for efficient pre-training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--smi_file",
        type=Path,
        required=True,
        help="Path to the input file containing cleaned SMILES strings.",
    )
    parser.add_argument(
        "--vocab_path",
        type=Path,
        required=True,
        help="Path to the fragment vocabulary file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Optional: Base directory for outputs. "
            "Defaults to the dataset's root directory (parent.parent of smi_file)."
        ),
    )
    parser.add_argument(
        "--order",
        type=int,
        default=1,
        help=(
            "Line-graph order k_line used by the featurizer.\n"
            "注意：如果 vocab 第一行 JSON 里已经写了 'line_order' 或 'order'，"
            "则以 vocab 为准；这个参数主要用于老 vocab 的兜底或显式 override。"
        ),
    )
    parser.add_argument(
        "--overlap_degree",
        type=int,
        default=None,
        help=(
            "k_overlap for fragment graph (min shared atoms to connect two fragments).\n"
            "如果不指定，则由 vocab meta 里的 'overlap_degree' 决定；"
            "若 meta 也没有，则默认 1。"
        ),
    )
    parser.add_argument(
        "--path_max_length",
        type=int,
        default=2,
        help="Maximum path length for the fragment graph featurization.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=16,
        help="Number of parallel CPU cores to use.",
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
            "Fragmentation method used inside the featurizer. "
            "Currently supports 'dove' and BRICS, r-BRICS variants.RECAP,RelMole"
        ),
    )

    args = parser.parse_args()

    if not args.smi_file.exists() or not args.vocab_path.exists():
        raise FileNotFoundError(
            f"Input SMILES file or vocabulary file not found: "
            f"{args.smi_file} / {args.vocab_path}"
        )

    if args.frag_method.startswith("accfg") and args.n_jobs > 1:
        print(
            f"[Warning] frag_method='{args.frag_method}' uses AccFG, which may "
            f"internally create its own process pool. "
            f"To avoid nested multiprocessing overhead, overriding n_jobs=1.",
            file=sys.stderr,
        )
        args.n_jobs = 1

    meta = load_vocab_meta(str(args.vocab_path))
    print("==== Vocab meta (from header JSON) ====")
    print(f"vocab_path    : {args.vocab_path}")
    print(f"frag_method   : {meta.frag_method}")
    print(f"line_order    : {meta.line_order}  (k_line)")
    print(f"overlap_degree: {meta.overlap_degree}  (k_overlap)")
    print(f"kekulize      : {meta.kekulize}")
    print("CLI overrides :")
    print(f"  --order          = {args.order}")
    print(f"  --overlap_degree = {args.overlap_degree}")
    print("========================================\n")

    if args.output_dir:
        output_base_dir = args.output_dir
    else:
        output_base_dir = args.smi_file.resolve().parent.parent

    lmdb_path = output_base_dir / "lmdb"
    final_smi_path = output_base_dir / "final.smi"

    with args.smi_file.open("r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    print("--- Starting LMDB Database Construction ---")
    print(f"  - Input SMILES: {len(smiles_list)} from {args.smi_file}")
    print(f"  - Vocabulary: {args.vocab_path}")
    print(f"  - Fragmentation method: {args.frag_method}")
    print(f"  - Output LMDB dir: {lmdb_path}")
    print(f"  - n_jobs (processes): {args.n_jobs}")

    lmdb_path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(lmdb_path), map_size=1099511627776)

    successful_count = 0
    final_smiles_list = []

    commit_every = 1000

    txn = env.begin(write=True)

    with Pool(
        args.n_jobs,
        initializer=_init_worker,
        initargs=(
            str(args.vocab_path),
            args.order,
            args.path_max_length,
            args.frag_method,
            args.overlap_degree,
        ),
    ) as pool:
        pbar = tqdm(
            pool.imap_unordered(_worker_featurize_and_serialize, smiles_list, chunksize=50),
            total=len(smiles_list),
            desc="Processing SMILES",
        )
        for smi, graph_bytes in pbar:
            if graph_bytes is None:
                continue

            key = str(successful_count).encode("utf-8")
            txn.put(key, graph_bytes)

            final_smiles_list.append(smi)
            successful_count += 1

            if successful_count % commit_every == 0:
                txn.commit()
                txn = env.begin(write=True)

    txn.commit()

    print(f"\nSuccessfully featurized {successful_count} / {len(smiles_list)} molecules.")

    if successful_count == 0:
        print("No molecules were successfully processed. Exiting.")
        env.close()
        return

    with env.begin(write=True) as txn_meta:
        txn_meta.put("num_examples".encode("utf-8"), str(successful_count).encode("utf-8"))
    env.close()

    final_smi_path.parent.mkdir(parents=True, exist_ok=True)
    with final_smi_path.open("w") as f:
        f.write("\n".join(final_smiles_list) + "\n")
    print(f"List of successfully processed SMILES saved to: {final_smi_path}")

    print(
        f"\n--- LMDB database construction complete! "
        f"Contains {successful_count} entries. ---"
    )


if __name__ == "__main__":
    main()
