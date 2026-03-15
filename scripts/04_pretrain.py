# scripts/04_pretrain.py
"""
Run the pre-training task for the CGMNet model.
"""

import argparse
import os
import random
import sys
from pathlib import Path
import warnings

warnings.filterwarnings(
    "ignore",
    category=Warning,
    module=r"pydantic\._internal\._generate_schema",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"torch\.utils\.data\.dataloader",
    message=r".*This DataLoader will create .* worker processes in total.*",
)

import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch.nn import BCEWithLogitsLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from cgmnet.utils.register import MODEL_REGISTRY, DATASET_REGISTRY, COLLATOR_REGISTRY
import cgmnet.data.dataset  # noqa: F401
import cgmnet.data.collator  # noqa: F401
import cgmnet.models.cgmnet  # noqa: F401

from cgmnet.training.pretrain_trainer import PretrainTrainer
from cgmnet.training.scheduler import PolynomialDecayLR
from cgmnet.utils.general_utils import set_random_seed
from configs.cgmnet_config import add_model_and_training_args


def setup_ddp():
    """Initialize the distributed environment."""
    if dist.is_initialized():
        return int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    return rank


def cleanup_ddp():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def seed_worker(worker_id):
    """Ensure that each data loader worker has a different random seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_vocab_size(vocab_path: Path) -> int:
    """
    Compute the vocabulary size from a vocab file and add one extra token
    for <unk>.
    """
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    with vocab_path.open("r") as f:
        header = f.readline()
        if not header:
            raise ValueError(f"Vocabulary file {vocab_path} is empty or corrupted.")
        num_smis = 0
        for line in f:
            if line.strip():
                num_smis += 1

    vocab_size = num_smis + 1
    print(f"[Vocab] Loaded from {vocab_path}: {num_smis} SMILES + 1 <unk> = {vocab_size} tokens")
    return vocab_size


def main():
    parser = argparse.ArgumentParser(description="Pre-train the CGMNet model.")

    parser.add_argument("--vocab_path", type=Path, required=True)
    parser.add_argument("--lmdb_path", type=Path, required=True)
    parser.add_argument("--descriptor_root", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)

    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--knodes", type=str, nargs="*", default=[])
    parser.add_argument("--n_steps", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-04)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--save_interval", type=int, default=25000)

    parser.add_argument("--n_threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--downsize", type=int, default=-1, help="Use only the first N samples of the dataset.")
    parser.add_argument(
        "--ddp_find_unused_parameters",
        action="store_true",
        help="Enable DDP find_unused_parameters flag. Use if you get DDP errors.",
    )

    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=3,
        help="Prefetch batches per worker when num_workers>0.",
    )
    parser.add_argument(
        "--no_persistent_workers",
        action="store_true",
        help="Disable persistent_workers (enabled by default if num_workers>0).",
    )

    parser.add_argument(
        "--tf32",
        action="store_true",
        help="Enable TF32 matmul (H100/A100) for faster training.",
    )

    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="CGMNet-Pretrain",
        help="Name of the WandB project.",
    )

    parser = add_model_and_training_args(parser)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    is_ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if is_ddp:
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    rank = setup_ddp() if is_ddp else 0
    world_size = dist.get_world_size() if is_ddp else 1
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    set_random_seed(args.seed + rank)

    if rank == 0:
        print("--- Pre-training Arguments ---")
        print(args)
        print(f"--- Running on {world_size} GPU(s) ---")
        if args.wandb:
            run_name = f"Pretrain-{args.save_path.parent.parent.name}-{args.save_path.name}"
            wandb.init(project=args.wandb_project, name=run_name, config=args)

    if rank == 0:
        print("Loading dataset...")
    dataset = DATASET_REGISTRY["cgmnet_pretrain"](
        lmdb_path=args.lmdb_path,
        descriptor_root=args.descriptor_root,
        knodes_to_load=args.knodes,
        downsize=args.downsize,
    )
    collator = COLLATOR_REGISTRY["cgmnet_pretrain"](mask_rate=args.mask_rate)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    ) if is_ddp else None

    use_persistent = (args.n_threads > 0) and (not args.no_persistent_workers)
    prefetch = args.prefetch_factor if args.n_threads > 0 else None

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.n_threads,
        worker_init_fn=seed_worker,
        collate_fn=collator,
        sampler=sampler,
        shuffle=(sampler is None),
        pin_memory=True,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch,
    )
    if rank == 0:
        print("Dataset loaded.")

    model_class = MODEL_REGISTRY["cgmnet"]
    n_tasks = get_vocab_size(args.vocab_path)
    model = model_class(args=args, n_tasks=n_tasks).to(device)

    if is_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=args.ddp_find_unused_parameters)

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model initialized with {num_params / 1e6:.3f}M parameters.")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = PolynomialDecayLR(
        optimizer,
        warmup_updates=2000,
        total_updates=args.n_steps,
        lr=args.lr,
        end_lr=1e-9,
        power=1.0,
    )

    loss_fn = BCEWithLogitsLoss(reduction="none")

    trainer = PretrainTrainer(args, optimizer, scheduler, loss_fn, device, ddp=is_ddp, rank=rank)

    trainer.fit(model, train_loader)

    if rank == 0 and args.wandb:
        wandb.finish()

    if is_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
