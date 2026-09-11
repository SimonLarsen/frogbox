import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)


class LRSchedulerWithWarmup(SequentialLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: LRScheduler,
        warmup_steps: int = 0,
        warmup_start_factor: float = 1.0e-7,
    ):
        schedulers: list[LRScheduler] = [scheduler]
        milestones = []

        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer=optimizer,
                start_factor=warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )

            schedulers.insert(0, warmup_scheduler)
            milestones.append(warmup_steps)

        super().__init__(optimizer, schedulers, milestones)


class CosineLRScheduler(LRSchedulerWithWarmup):
    """Cosine annealing LR scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iterations: int,
        end_value: float = 1.0e-7,
        warmup_steps: int = 0,
        warmup_start_factor: float = 1.0e-7,
    ):
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=max_iterations - warmup_steps,
            eta_min=end_value,
        )

        super().__init__(optimizer, scheduler, warmup_steps, warmup_start_factor)


class LinearLRScheduler(LRSchedulerWithWarmup):
    """Linear LR scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iterations: int,
        end_value: float = 1.0e-7,
        warmup_steps: int = 0,
        warmup_start_factor: float = 1.0e-7,
    ):
        start_value = optimizer.param_groups[0]["lr"]

        scheduler = LinearLR(
            optimizer=optimizer,
            start_factor=1.0,
            end_factor=end_value / start_value,
            total_iters=max_iterations - warmup_steps,
        )

        super().__init__(optimizer, scheduler, warmup_steps, warmup_start_factor)
