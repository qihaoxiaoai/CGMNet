# cgmnet/training/finetune_trainer.py
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cgmnet.models.utils import ModelWithEMA


def _to_jsonable(x):
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]

    if isinstance(x, Path):
        return str(x)
    if isinstance(x, torch.device):
        return str(x)

    if isinstance(x, torch.Tensor):
        if x.ndim == 0:
            return x.item()
        return x.detach().cpu().tolist()

    try:
        import numpy as np  # noqa: F401

        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
    except Exception:
        pass

    if hasattr(x, "item") and callable(getattr(x, "item")):
        try:
            return x.item()
        except Exception:
            pass
    if hasattr(x, "tolist") and callable(getattr(x, "tolist")):
        try:
            return x.tolist()
        except Exception:
            pass

    return x


class ResultTracker:
    def __init__(self, metric_name: str):
        self.metric_name = metric_name
        self.lower_is_better = metric_name in ["rmse", "mae"]
        self.best_score = float("inf") if self.lower_is_better else -float("inf")
        self.best_epoch = -1

    def update(self, score: float, epoch: int) -> bool:
        improved = (score < self.best_score) if self.lower_is_better else (score > self.best_score)
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            return True
        return False


class FinetuneTrainer:
    def __init__(self, args, model, optimizer, scheduler, loss_fn, evaluator, device, main_metric: str):
        self.args = args
        self.model = ModelWithEMA(model, decay=args.ema_decay)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.evaluator = evaluator
        self.device = device

        self.save_dir = Path(args.save_path)
        self.main_metric = main_metric
        self.valid_tracker = ResultTracker(main_metric)

        self.target_mean = getattr(args, "target_mean", None)
        self.target_std = getattr(args, "target_std", None)

        self.no_save_model = getattr(args, "no_save_model", False)
        if not self.no_save_model:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            print("[FinetuneTrainer] no_save_model=True, will NOT write any .pth checkpoints.", flush=True)

        self._wandb = None
        if getattr(self.args, "wandb", False):
            try:
                import wandb  # noqa: F401

                self._wandb = wandb
            except Exception as e:
                raise RuntimeError("args.wandb=True but wandb import failed. Please install wandb.") from e

        self.early_stop_patience = int(getattr(args, "early_stop_patience", 0))
        self.early_stop_min_delta = float(getattr(args, "early_stop_min_delta", 0.0))

        self.freeze_encoder_epochs = int(getattr(args, "freeze_encoder_epochs", 0))

        self._best_ema_state_cpu = None
        self._encoder_is_frozen = False

    def _base_model(self):
        # ModelWithEMA typically wraps the model in .model and stores EMA in .ema_model
        return getattr(self.model, "model", self.model)

    def _set_encoder_trainable(self, trainable: bool):
        base = self._base_model()
        changed = False
        for name, p in base.named_parameters():
            if name.startswith("predictor."):
                continue
            if p.requires_grad != trainable:
                p.requires_grad = trainable
                changed = True

        self._encoder_is_frozen = not trainable
        if changed:
            state = "UNFROZEN" if trainable else "FROZEN"
            print(f"[Freeze] Encoder is now {state}.", flush=True)

    def _maybe_apply_freeze_schedule(self, epoch: int):
        if self.freeze_encoder_epochs <= 0:
            return
        if epoch <= self.freeze_encoder_epochs:
            if not self._encoder_is_frozen:
                self._set_encoder_trainable(False)
        else:
            if self._encoder_is_frozen:
                self._set_encoder_trainable(True)

    def _move_batch_to_device(self, batched_data, labels):
        for key, value in batched_data.items():
            if hasattr(value, "to"):
                batched_data[key] = value.to(self.device)
            elif isinstance(value, dict):
                batched_data[key] = {k: v.to(self.device) for k, v in value.items()}
        return batched_data, labels.to(self.device)

    def _maybe_denormalize_preds(self, preds_cpu: torch.Tensor) -> torch.Tensor:
        if self.target_mean is None or self.target_std is None:
            return preds_cpu
        mean = torch.as_tensor(self.target_mean, dtype=preds_cpu.dtype, device=preds_cpu.device)
        std = torch.as_tensor(self.target_std, dtype=preds_cpu.dtype, device=preds_cpu.device)
        return preds_cpu * std + mean

    def _evaluate_epoch(self, dataloader, model_to_eval):
        model_to_eval.eval()

        all_preds = []
        all_labels = []

        total_loss = 0.0
        total_labeled = 0

        with torch.no_grad():
            for batched_data, labels in dataloader:
                if batched_data is None:
                    continue

                batched_data, labels = self._move_batch_to_device(batched_data, labels)
                preds = model_to_eval(batched_data)

                is_labeled = torch.isfinite(labels)
                valid_labels = torch.nan_to_num(labels)

                labeled_count = int(is_labeled.sum().item())
                if labeled_count > 0:
                    loss = (self.loss_fn(preds, valid_labels) * is_labeled).sum() / is_labeled.sum()
                    total_loss += float(loss.item()) * labeled_count
                    total_labeled += labeled_count

                all_preds.append(preds.detach().cpu())
                all_labels.append(labels.detach().cpu())

        avg_loss = (total_loss / total_labeled) if total_labeled > 0 else 0.0
        if not all_preds:
            return {"nan": float("nan")}, avg_loss

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_preds = self._maybe_denormalize_preds(all_preds)

        scores = self.evaluator.eval(all_labels, all_preds)
        scores = _to_jsonable(scores)
        return scores, avg_loss

    def _train_epoch(self, train_loader: DataLoader, epoch: int):
        # Apply freeze/unfreeze before starting the epoch
        self._maybe_apply_freeze_schedule(epoch)

        self.model.train()

        total_loss = 0.0
        total_labeled = 0

        data_iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{self.args.epochs}")
        for batched_data, labels in data_iterator:
            if batched_data is None:
                continue

            batched_data, labels = self._move_batch_to_device(batched_data, labels)

            self.optimizer.zero_grad(set_to_none=True)
            preds = self.model(batched_data)

            is_labeled = torch.isfinite(labels)
            train_labels = torch.nan_to_num(labels)

            if self.target_mean is not None and self.target_std is not None:
                mean = torch.as_tensor(self.target_mean, device=self.device, dtype=train_labels.dtype)
                std = torch.as_tensor(self.target_std, device=self.device, dtype=train_labels.dtype)
                train_labels = (train_labels - mean) / std

            labeled_count = int(is_labeled.sum().item())
            if labeled_count == 0:
                continue

            loss = (self.loss_fn(preds, train_labels) * is_labeled).sum() / is_labeled.sum()
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()

            # Clip only trainable params (requires_grad=True)
            params_to_clip = [p for p in self._base_model().parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)

            self.optimizer.step()
            self.model.update_ema()
            self.scheduler.step()

            loss_val = float(loss.item())
            total_loss += loss_val * labeled_count
            total_labeled += labeled_count

            data_iterator.set_postfix(loss=f"{loss_val:.4f}")
            if self._wandb is not None:
                self._wandb.log({"train_step_loss": loss_val, "lr": self.scheduler.get_last_lr()[0]})

        return (total_loss / total_labeled) if total_labeled > 0 else 0.0

    def _dump_run_info(self):
        if self.no_save_model:
            return

        info_path = self.save_dir / "run_info.json"
        try:
            cfg = _to_jsonable(vars(self.args))
            meta = {
                "dataset": self.args.dataset,
                "scaffold_id": self.args.scaffold_id,
                "seed": self.args.seed,
                "main_metric": self.main_metric,
                "best_score": _to_jsonable(self.valid_tracker.best_score),
                "best_epoch": int(self.valid_tracker.best_epoch),
                "config": cfg,
            }
            with open(info_path, "w") as f:
                json.dump(_to_jsonable(meta), f, indent=2)
        except Exception as e:
            print(f"[WARN] failed to write run_info.json: {e}")

    def _save_test_results(self, test_scores: dict, test_loss: float | None):
        if self.no_save_model:
            return

        out = {
            "main_metric": self.main_metric,
            "best_valid_score": _to_jsonable(self.valid_tracker.best_score),
            "best_valid_epoch": int(self.valid_tracker.best_epoch),
            "test_loss": _to_jsonable(test_loss),
            "test_scores": _to_jsonable(test_scores),
        }
        try:
            with open(self.save_dir / "test_results.json", "w") as f:
                json.dump(_to_jsonable(out), f, indent=2)
        except Exception as e:
            print(f"[WARN] failed to write test_results.json: {e}")

    def fit(self, train_loader, valid_loader, test_loader):
        print("Starting fine-tuning...")

        best_ckpt_path = self.save_dir / "best_model.pth"
        epochs_no_improve = 0

        # If freeze is enabled, start frozen before the first epoch
        if self.freeze_encoder_epochs > 0:
            self._set_encoder_trainable(False)

        for epoch in range(1, self.args.epochs + 1):
            avg_train_loss = self._train_epoch(train_loader, epoch)

            valid_scores, valid_loss = self._evaluate_epoch(valid_loader, self.model.ema_model)
            current_valid = valid_scores.get(self.main_metric, float("nan"))

            valid_scores_str = ", ".join(
                [f"{k.upper()}={float(v):.4f}" for k, v in valid_scores.items() if isinstance(v, (float, int))]
            )
            print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Valid Loss={valid_loss:.4f} | {valid_scores_str}")

            if self._wandb is not None:
                self._wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": avg_train_loss,
                        "valid_loss": valid_loss,
                        **{f"valid_{k}": _to_jsonable(v) for k, v in valid_scores.items()},
                    }
                )

            improved = self.valid_tracker.update(float(current_valid), epoch)

            if improved:
                epochs_no_improve = 0
                if not self.no_save_model:
                    print(
                        f"*** New best validation {self.main_metric.upper()}: {float(current_valid):.4f}. Saving best_model.pth ***"
                    )
                    torch.save(self.model.ema_model.state_dict(), best_ckpt_path)
                    self._dump_run_info()
                else:
                    self._best_ema_state_cpu = {k: v.detach().cpu() for k, v in self.model.ema_model.state_dict().items()}
                    print(
                        f"*** New best validation {self.main_metric.upper()}: {float(current_valid):.4f} "
                        f"(checkpoint saving disabled) ***"
                    )
            else:
                epochs_no_improve += 1
                if self.early_stop_patience > 0:
                    if epochs_no_improve >= self.early_stop_patience:
                        print(f"[EarlyStop] No improvement for {self.early_stop_patience} epochs. Stopping at epoch {epoch}.")
                        break

        test_scores = None
        test_loss = None

        if test_loader is not None:
            if (not self.no_save_model) and best_ckpt_path.exists():
                state = torch.load(best_ckpt_path, map_location=self.device)
                self.model.ema_model.load_state_dict(state)
            elif self.no_save_model and self._best_ema_state_cpu is not None:
                state = {k: v.to(self.device) for k, v in self._best_ema_state_cpu.items()}
                self.model.ema_model.load_state_dict(state)

            test_scores, test_loss = self._evaluate_epoch(test_loader, self.model.ema_model)

            test_str = ", ".join(
                [f"{k.upper()}={float(v):.4f}" for k, v in test_scores.items() if isinstance(v, (float, int))]
            )
            print(f"[TEST] {test_str} | Loss={float(test_loss):.4f}")

            if self._wandb is not None:
                best_ep = self.valid_tracker.best_epoch if self.valid_tracker.best_epoch >= 0 else 0
                self._wandb.log(
                    {
                        "epoch": best_ep,
                        "test_loss": float(test_loss),
                        **{f"test_{k}": _to_jsonable(v) for k, v in test_scores.items()},
                    }
                )
                if self._wandb.run is not None:
                    self._wandb.run.summary["best_valid_epoch"] = int(self.valid_tracker.best_epoch)
                    self._wandb.run.summary[f"best_valid_{self.main_metric}"] = _to_jsonable(self.valid_tracker.best_score)
                    for k, v in test_scores.items():
                        self._wandb.run.summary[f"test_{k}"] = _to_jsonable(v)
                    self._wandb.run.summary["test_loss"] = float(test_loss)

            self._save_test_results(test_scores, test_loss)

        msg = (
            f"\nFine-tuning finished.\n"
            f"Best VALID {self.main_metric.upper()}: {float(self.valid_tracker.best_score):.4f} "
            f"at epoch {int(self.valid_tracker.best_epoch)}\n"
        )
        if test_scores is not None and self.main_metric in test_scores:
            msg += f"TEST ({self.main_metric.upper()}) on best-valid checkpoint: {float(test_scores[self.main_metric]):.4f}\n"
        print(msg)

