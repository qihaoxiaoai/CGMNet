# cgmnet/training/pretrain_trainer.py
"""
Defines the Trainer class for the pre-training task.
"""
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from tqdm import tqdm
import wandb


class PretrainTrainer:
    """
    Handles the pre-training loop, including forward pass, backward pass,
    optimization, and checkpointing.
    """
    def __init__(
        self,
        args,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        loss_fn,
        device: torch.device,
        ddp: bool = False,
        rank: int = 0
    ):
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.ddp = ddp
        self.rank = rank
        self.n_updates = 0
        self.save_path = Path(args.save_path)
        if self.rank == 0:
            self.save_path.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, model, train_loader, current_epoch):
        """Trains the model for one epoch."""
        model.train()
        data_iterator = tqdm(
            train_loader,
            desc=f"Epoch {current_epoch} (Steps: {self.n_updates})",
            disable=(self.rank != 0)
        )

        for batched_data, labels in data_iterator:
            if batched_data is None:
                continue

            # move to device
            for key, value in batched_data.items():
                if hasattr(value, 'to'):
                    batched_data[key] = value.to(self.device)
                elif isinstance(value, dict):
                    batched_data[key] = {k: v.to(self.device) for k, v in value.items()}
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward: model 根据 training/predictor 状态自己决定是 pretrain 还是 finetune
            logits = model(batched_data)  # [num_masked, n_tasks]

            # === 构造 true_labels ===
            #
            # 逻辑：
            #   - 有 hier 信息 且 没有设置 --no_hier_loss：
            #         → 用 hier_row_idx / hier_col_idx 构造多标签 multi-hot
            #   - 否则：
            #         → 回退到原来的单标签 one-hot（labels 只有一个 id）
            if (not self.args.no_hier_loss) and \
               ('hier_row_idx' in batched_data) and \
               ('hier_col_idx' in batched_data) and \
               (batched_data['hier_row_idx'].numel() > 0):

                # hierarchical 多标签：true_labels[i, j] = 1 表示第 i 个 masked fragment
                # 对应的 nested fragment 集合里包含 vocab id = j
                true_labels = torch.zeros_like(logits, device=self.device)

                row_idx = batched_data['hier_row_idx'].long()
                col_idx = batched_data['hier_col_idx'].long()

                # 安全过滤一下越界的 col（以防 vocab / n_tasks 不一致）
                num_rows, num_cols = logits.shape
                valid_mask = (row_idx >= 0) & (row_idx < num_rows) & \
                             (col_idx >= 0) & (col_idx < num_cols)
                if valid_mask.any():
                    row_idx = row_idx[valid_mask]
                    col_idx = col_idx[valid_mask]
                    true_labels[row_idx, col_idx] = 1.0
                # 如果全被过滤掉，就退化成全 0，loss 会接近 0，训练相当于空步
            else:
                # 原始 MCP 单标签 one-hot 分支
                # labels: [num_masked] 或 [num_masked, 1]
                if labels.ndim == 1:
                    labels = labels.unsqueeze(-1)

                true_labels = torch.zeros_like(logits, device=self.device)
                rows = torch.arange(logits.size(0), device=self.device).repeat_interleave(labels.size(1))
                cols = labels.flatten()

                # 同样做一下合法性过滤（防止偶发越界）
                valid_mask = (cols >= 0) & (cols < logits.size(1))
                if valid_mask.any():
                    rows = rows[valid_mask]
                    cols = cols[valid_mask]
                    true_labels[rows, cols] = 1.0

            # === 计算 loss ===
            # 建议 loss_fn 用 BCEWithLogitsLoss(reduction='none')
            loss = self.loss_fn(logits, true_labels).mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            self.n_updates += 1

            if self.rank == 0:
                lr = self.scheduler.get_last_lr()[0]
                data_iterator.set_postfix(loss=loss.item(), lr=f"{lr:.2e}")
                if self.args.wandb:
                    wandb.log({"train_loss": loss.item(), "lr": lr, "step": self.n_updates})

            if self.n_updates >= self.args.n_steps:
                break

            if self.rank == 0 and self.n_updates % self.args.save_interval == 0:
                self.save_checkpoint(model)

    def fit(self, model, train_loader):
        """Main training loop."""
        if self.rank == 0:
            print(f"Starting pre-training, targeting {self.args.n_steps} steps...")

        # In a step-based training, we loop until steps are met.
        # A simple epoch count is just a guide.
        current_epoch = 1
        while self.n_updates < self.args.n_steps:
            self.train_epoch(model, train_loader, current_epoch)
            current_epoch += 1

        if self.rank == 0:
            print("\nTarget steps reached. Saving final model.")
            self.save_checkpoint(model, is_final=True)

    def save_checkpoint(self, model, is_final=False):
        """Saves a model checkpoint."""
        model_to_save = model.module if self.ddp else model
        suffix = "final" if is_final else f"step_{self.n_updates}"
        ckpt_path = self.save_path / f"model_{suffix}.pth"
        torch.save(model_to_save.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

