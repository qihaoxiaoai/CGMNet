# scripts/06_finetune.py
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

warnings.filterwarnings(
    "ignore",
    message=r"The 'repr' attribute with value False was provided to the `Field\(\)` function.*",
    module=r"pydantic\._internal\._generate_schema",
)
warnings.filterwarnings(
    "ignore",
    message=r"The 'frozen' attribute with value True was provided to the `Field\(\)` function.*",
    module=r"pydantic\._internal\._generate_schema",
)

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from cgmnet.utils.register import MODEL_REGISTRY, DATASET_REGISTRY, COLLATOR_REGISTRY

import cgmnet.data.dataset
import cgmnet.data.collator
import cgmnet.models.cgmnet

from cgmnet.data.featurizer import CGMNetFeaturizer
from cgmnet.data.fragmentizer import load_vocab_meta
from cgmnet.training.evaluator import Evaluator
from cgmnet.training.finetune_trainer import FinetuneTrainer
from cgmnet.training.scheduler import WarmupCosineDecayLR
from cgmnet.utils.data_utils import get_task_config
from cgmnet.utils.general_utils import set_random_seed
from configs.cgmnet_config import add_model_and_training_args


def _materialize_save_path_with_run_id(save_path: Path, run_id: str | None) -> Path:
    save_path = Path(os.path.expandvars(str(save_path)))
    if run_id:
        return save_path if save_path.name == run_id else (save_path / run_id)
    ts = time.strftime("run_%Y%m%d_%H%M%S")
    return save_path if save_path.name == ts else (save_path / ts)


def _move_batch_to_device(batched_data, labels, device: torch.device):
    for key, value in batched_data.items():
        if hasattr(value, "to"):
            batched_data[key] = value.to(device)
        elif isinstance(value, dict):
            batched_data[key] = {k: v.to(device) for k, v in value.items()}
    return batched_data, labels.to(device)


def _is_head_key(k: str) -> bool:
    return k.startswith("pretrain_predictor.") or k.startswith("predictor.")


def _load_backbone_only(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    print(f"[CKPT] Loading pre-trained model from: {ckpt_path}", flush=True)
    state_dict = torch.load(ckpt_path, map_location=device)

    if len(state_dict) > 0 and next(iter(state_dict)).startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model_dict = model.state_dict()

    loaded = {}
    skipped = []
    skipped_head = []

    for k, v in state_dict.items():
        if k not in model_dict:
            skipped.append(k)
            continue
        if _is_head_key(k):
            skipped_head.append(k)
            continue
        if model_dict[k].shape != v.shape:
            skipped.append(k)
            continue
        loaded[k] = v

    model_dict.update(loaded)
    model.load_state_dict(model_dict)

    print(
        f"[CKPT] Loaded backbone-only. loaded={len(loaded)}, skipped={len(skipped)}, skipped_head={len(skipped_head)}",
        flush=True,
    )


def _maybe_zero_init_finetune_head(model: torch.nn.Module):
    """
    Optional stabilization:
    set last Linear layer of predictor to zeros, so initial preds ~0 in STDZ space.
    """
    if getattr(model, "predictor", None) is None:
        return
    last = None
    for m in model.predictor.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is not None:
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)
        print("[INIT] Zero-initialized finetune head last layer (predictor output ~0 at init).", flush=True)


def _init_sanity_check(
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    target_mean: float | None,
    target_std: float | None,
    max_batches: int = 1,
):
    """
    Prints init stats to diagnose scale/bias and label standardization.

    In trainer, regression labels are standardized before loss:
        y_stdz = (y_raw - mean) / std
    Model preds during finetune are in the SAME standardized space.
    """
    model.eval()

    preds_all = []
    labels_all = []

    with torch.no_grad():
        n = 0
        for batched_data, labels in train_loader:
            if batched_data is None:
                continue
            batched_data, labels = _move_batch_to_device(batched_data, labels, device)

            preds = model(batched_data)
            preds_all.append(preds.detach().cpu())
            labels_all.append(labels.detach().cpu())

            n += 1
            if n >= max_batches:
                break

    if not preds_all:
        print("[INIT] no batches available for init sanity check.", flush=True)
        return

    preds = torch.cat(preds_all, dim=0).view(-1)
    labels = torch.cat(labels_all, dim=0).view(-1)

    is_lab = torch.isfinite(labels)
    labels_f = labels[is_lab]
    preds_f = preds[is_lab]

    if labels_f.numel() == 0:
        print("[INIT] no finite labels found in init sanity check.", flush=True)
        return

    lab_mean = labels_f.mean().item()
    lab_std = labels_f.std(unbiased=False).item()

    print(f"[INIT STAT] labels mean/std (RAW)  = {lab_mean:.4f} / {lab_std:.4f}", flush=True)
    print(
        f"[INIT STAT] preds  mean/std (STDZ) = {preds_f.mean().item():.4f} / {preds_f.std(unbiased=False).item():.4f}",
        flush=True,
    )

    if target_mean is not None and target_std is not None:
        labels_stdz = (labels_f - float(target_mean)) / float(target_std)
        rmse_stdz = torch.sqrt(torch.mean((preds_f - labels_stdz) ** 2)).item()

        preds_raw = preds_f * float(target_std) + float(target_mean)
        rmse_raw = torch.sqrt(torch.mean((preds_raw - labels_f) ** 2)).item()

        print(
            f"[INIT STAT] labels mean/std (STDZ) = {labels_stdz.mean().item():.4f} / {labels_stdz.std(unbiased=False).item():.4f}",
            flush=True,
        )
        print(f"[INIT RMSE] RMSE in STDZ space = {rmse_stdz:.4f}", flush=True)
        print(f"[INIT RMSE] RMSE in RAW  space = {rmse_raw:.4f}", flush=True)
    else:
        print("[INIT] target_mean/std unavailable; skip label-standardized RMSE.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune CGMNet with hybrid task configuration.")

    group = parser.add_argument_group("Path Arguments")
    group.add_argument("--model_path", type=Path, required=True, help="Path to pre-trained model.")
    group.add_argument("--dataset", type=str, required=True, help="Name of the fine-tuning dataset.")
    group.add_argument("--dataset_root", type=Path, required=True, help="Root directory of datasets.")
    group.add_argument("--vocab_path", type=Path, required=True, help="Path to the vocabulary file.")
    group.add_argument("--save_path", type=Path, required=True, help="Directory to save outputs/checkpoints.")

    group = parser.add_argument_group("Fine-tuning Hyperparameters")
    group.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    group.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs.")
    group.add_argument("--lr", type=float, default=3e-5, help="Base learning rate for encoder.")
    group.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    group.add_argument("--batch_size", type=int, default=32, help="Batch size.")

    group = parser.add_argument_group("Freeze / Head LR Options")
    group.add_argument(
        "--freeze_encoder_epochs",
        type=int,
        default=0,
        help="Freeze encoder for first N epochs (train predictor only). Default 0 keeps original behavior.",
    )
    group.add_argument(
        "--head_lr_mult",
        type=float,
        default=1.0,
        help="Multiplier for predictor learning rate. Default 1.0 keeps original behavior.",
    )

    group = parser.add_argument_group("Init Debug Options")
    group.add_argument("--debug_init_only", action="store_true", help="Run init sanity check then exit.")
    group.add_argument("--init_check_batches", type=int, default=1, help="How many train batches for init check.")
    group.add_argument(
        "--zero_init_head",
        action="store_true",
        help="Zero-init finetune head last layer (predictor output ~0 at init).",
    )

    group = parser.add_argument_group("System, Logging & Misc")
    group.add_argument("--scaffold_id", type=int, default=0, help="Scaffold split ID.")
    group.add_argument("--device_id", type=int, default=0, help="GPU device id.")
    group.add_argument("--n_threads", type=int, default=1, help="Number of CPU workers for data loading.")
    group.add_argument("--seed", type=int, default=42, help="Random seed.")
    group.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    group.add_argument("--wandb_project", type=str, default="CGMNet-Finetune", help="WandB project name.")
    group.add_argument("--knodes", type=str, nargs="*", default=[], help="Knowledge node feature names.")
    group.add_argument("--no_save_model", action="store_true", help="If set, do NOT save any finetuned model checkpoints.")

    group.add_argument("--order", type=int, default=1, help="Fallback line-graph order (k_line) if not present in vocab metadata.")

    group = parser.add_argument_group("Fragmentization")
    group.add_argument(
        "--frag_method",
        type=str,
        default="dove",
        choices=[
            "dove",
            "brics",
            "brics_vanilla",
            "brics_overlap",
            "rbrics",
            "rbrics_vanilla",
            "rbrics_overlap",
            "recap",
            "recap_vanilla",
            "recap_overlap",
            "relmole",
            "relmole_vanilla",
            "relmole_overlap",
            "accfg",
            "accfg_vanilla",
            "accfg_overlap",
            "jt",
            "jt_vae",
            "macfrag",
            "macfrag_vanilla",
            "macfrag_overlap",
            "bm",
            "bm_vanilla",
            "bm_overlap",
            "efgs",
            "efgs_vanilla",
            "efgs_overlap",
            "louvain",
        ],
        help="Fragmentation method (must match pretraining/LMDB build).",
    )
    group.add_argument("--overlap_degree", type=int, default=None, help="k_overlap for fragment graph; if None, try reading from vocab metadata.")

    group = parser.add_argument_group("Graph Cache (optional)")
    group.add_argument("--cache_graphs", action="store_true", help="Enable LMDB graph cache for featurizer outputs.")
    group.add_argument("--cache_dir", type=Path, default=None, help="Cache directory.")

    parser = add_model_and_training_args(parser)
    args = parser.parse_args()

    set_random_seed(args.seed)
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")

    task_config = get_task_config(dataset_name=args.dataset, dataset_root=args.dataset_root)
    task_type = task_config["task_type"]
    n_tasks = task_config["n_tasks"]
    metrics = task_config["metrics"]
    main_metric = metrics[0]

    wandb = None
    run_id = os.environ.get("WANDB_RUN_ID")

    if args.wandb:
        try:
            import wandb as _wandb

            wandb = _wandb
            wandb.init(
                project=args.wandb_project,
                name=f"{args.dataset}-{args.frag_method}-scaffold{args.scaffold_id}-seed{args.seed}",
                config=vars(args),
            )
            if wandb.run is not None:
                run_id = getattr(wandb.run, "id", run_id)
        except Exception as e:
            print(f"[WARN] WandB disabled (import/init failed): {e}", file=sys.stderr)
            args.wandb = False
            wandb = None

    args.save_path = _materialize_save_path_with_run_id(args.save_path, run_id)
    args.save_path.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Models will be saved to: {args.save_path.resolve()}")

    vocab_meta = None
    try:
        vocab_meta = load_vocab_meta(str(args.vocab_path))
        print("==== Vocab metadata ====")
        print(f"vocab_path     : {args.vocab_path}")
        print(f"frag_method    : {vocab_meta.frag_method}")
        print(f"line_order     : {vocab_meta.line_order}")
        print(f"overlap_degree : {vocab_meta.overlap_degree}")
        print(f"kekulize       : {vocab_meta.kekulize}")
        print("========================")
    except Exception as e:
        print(f"[WARN] Failed to load vocab metadata from {args.vocab_path}: {e}", file=sys.stderr)

    if args.cache_graphs:
        if args.cache_dir is None:
            base_cache_root = Path(args.dataset_root) / "graph_cache"
            if vocab_meta is not None:
                frag_tag = vocab_meta.frag_method or args.frag_method
                k_line = vocab_meta.line_order if vocab_meta.line_order is not None else args.order
                eff_overlap = vocab_meta.overlap_degree
            else:
                frag_tag = args.frag_method
                k_line = args.order
                eff_overlap = args.overlap_degree

            cache_name = f"{args.dataset}_{frag_tag}_k{k_line}"
            if eff_overlap is not None:
                cache_name += f"_ov{eff_overlap}"
            cache_name += f"_p{args.path_max_length}_{args.vocab_path.stem}"
            args.cache_dir = base_cache_root / cache_name

        print(f"[GraphCache] Enabled. Cache dir = {args.cache_dir}", flush=True)
    else:
        if args.cache_dir is not None:
            print(f"[GraphCache] cache_graphs=False, ignoring cache_dir={args.cache_dir}", file=sys.stderr)

    print(f"Loading dataset: {args.dataset}, Task: {task_type}, Main Metric: {main_metric.upper()}")

    DatasetClass = DATASET_REGISTRY["cgmnet_finetune"]
    CollatorClass = COLLATOR_REGISTRY["cgmnet_finetune"]
    ModelClass = MODEL_REGISTRY["cgmnet"]

    featurizer = CGMNetFeaturizer(
        vocab_path=args.vocab_path,
        order=args.order,
        max_path_length=args.path_max_length,
        frag_method=args.frag_method,
        overlap_degree=args.overlap_degree,
    )

    ds_kwargs = dict(
        dataset_name=args.dataset,
        dataset_root=args.dataset_root,
        featurizer=featurizer,
        knodes_to_load=args.knodes,
        scaffold_id=args.scaffold_id,
        cache_graphs=args.cache_graphs,
        cache_dir=args.cache_dir,
    )

    train_dataset = DatasetClass(split="train", **ds_kwargs)
    valid_dataset = DatasetClass(split="valid", **ds_kwargs)
    test_dataset = DatasetClass(split="test", **ds_kwargs)

    collator = CollatorClass()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.n_threads,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.n_threads,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.n_threads,
        pin_memory=True,
    )

    if task_type == "reg" and hasattr(train_dataset, "target_mean"):
        args.target_mean = [float(x) for x in train_dataset.target_mean.detach().cpu().view(-1).tolist()]
        args.target_std = [float(x) for x in train_dataset.target_std.detach().cpu().view(-1).tolist()]
        print("-" * 50)
        print(f"Regression normalization constants for '{args.dataset}':")
        print(f"  target_mean: {float(args.target_mean[0]):.6f}")
        print(f"  target_std : {float(args.target_std[0]):.6f}")
        print("-" * 50)

    model = ModelClass(args=args, n_tasks=n_tasks).to(device)

    model.init_ft_predictor(n_tasks=n_tasks, dropout=args.finetune_dropout)

    _load_backbone_only(model=model, ckpt_path=args.model_path, device=device)

    if args.zero_init_head:
        _maybe_zero_init_finetune_head(model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {num_params / 1e6:.3f}M trainable parameters.", flush=True)

    if task_type == "reg":
        tmean = float(args.target_mean[0]) if hasattr(args, "target_mean") and args.target_mean is not None else None
        tstd = float(args.target_std[0]) if hasattr(args, "target_std") and args.target_std is not None else None
        _init_sanity_check(
            model=model,
            train_loader=train_loader,
            device=device,
            target_mean=tmean,
            target_std=tstd,
            max_batches=int(args.init_check_batches),
        )
        if args.debug_init_only:
            print("[DEBUG] Exit after init sanity check.", flush=True)
            return

    encoder_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("predictor."):
            head_params.append(p)
        else:
            encoder_params.append(p)

    head_lr = args.lr * float(args.head_lr_mult)
    print(f"[OPT] encoder_lr={args.lr:.6g}, head_lr={head_lr:.6g}, freeze_encoder_epochs={int(args.freeze_encoder_epochs)}")

    optimizer = Adam(
        [
            {"params": encoder_params, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": head_params, "lr": head_lr, "weight_decay": args.weight_decay},
        ]
    )

    scheduler = WarmupCosineDecayLR(
        optimizer,
        total_iters=len(train_loader) * args.epochs,
        warmup_iters=len(train_loader) * args.warmup_epochs,
    )

    loss_fn = nn.BCEWithLogitsLoss(reduction="none") if task_type == "cls" else nn.MSELoss(reduction="none")
    evaluator = Evaluator(metrics=metrics, n_tasks=n_tasks)

    trainer = FinetuneTrainer(args, model, optimizer, scheduler, loss_fn, evaluator, device, main_metric)
    trainer.fit(train_loader, valid_loader, test_loader)

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
