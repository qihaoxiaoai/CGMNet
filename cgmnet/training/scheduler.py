# cgmnet/training/scheduler.py
"""
Learning rate schedulers for model training.
"""
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class PolynomialDecayLR(_LRScheduler):
    """
    A learning rate scheduler that implements a polynomial decay schedule with a
    warmup phase.

    During the warmup phase, the learning rate increases linearly from 0 to the
    target learning rate. After warmup, the learning rate decays polynomially
    towards a specified end learning rate.

    Args:
        optimizer (Optimizer): The optimizer wrapped by the scheduler.
        warmup_updates (int): The number of steps for the warmup phase.
        total_updates (int): The total number of training steps.
        lr (float): The target learning rate after warmup.
        end_lr (float): The final learning rate after decay.
        power (float): The exponent for the polynomial decay.
    """
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_updates: int,
                 total_updates: int,
                 lr: float,
                 end_lr: float,
                 power: float,
                 last_epoch: int = -1):
        self.warmup_updates = warmup_updates
        self.total_updates = total_updates
        self.initial_lr = lr
        self.end_lr = end_lr
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculates the learning rate for the current step."""
        current_step = self._step_count
        if current_step <= self.warmup_updates:
            # Linear warmup
            warmup_factor = current_step / float(self.warmup_updates)
            lr = self.initial_lr * warmup_factor
        elif current_step >= self.total_updates:
            # Reached the end
            lr = self.end_lr
        else:
            # Polynomial decay
            progress = (current_step - self.warmup_updates) / \
                       (self.total_updates - self.warmup_updates)
            
            lr_range = self.initial_lr - self.end_lr
            lr = lr_range * ((1 - progress) ** self.power) + self.end_lr

        return [lr for _ in self.optimizer.param_groups]


class WarmupCosineDecayLR(_LRScheduler):
    """
    A learning rate scheduler that implements a linear warmup followed by a
    cosine decay schedule.

    Args:
        optimizer (Optimizer): The optimizer wrapped by the scheduler.
        total_iters (int): The total number of training steps.
        warmup_iters (int): The number of steps for the linear warmup phase.
    """
    def __init__(self, 
                 optimizer: Optimizer, 
                 total_iters: int, 
                 warmup_iters: int = 0, 
                 last_epoch: int = -1):
        self.total_iters = total_iters
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculates the learning rate for the current step."""
        current_step = self.last_epoch
        lrs = []
        for base_lr in self.base_lrs:
            if current_step < self.warmup_iters:
                # Linear warmup
                factor = current_step / self.warmup_iters if self.warmup_iters > 0 else 1.0
                lr = base_lr * factor
            else:
                # Cosine decay phase
                progress = (current_step - self.warmup_iters) / (self.total_iters - self.warmup_iters)
                cos_decay = 0.5 * (1 + math.cos(math.pi * progress))
                lr = base_lr * cos_decay
            lrs.append(lr)
        return lrs
