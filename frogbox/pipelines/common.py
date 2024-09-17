from typing import Dict, Union
import torch
from ..config import (
    LossDefinition,
    SchedulerType,
    LRSchedulerDefinition,
    create_object_from_config,
)
from .composite_loss import CompositeLoss


def create_composite_loss(
    config: Dict[str, LossDefinition],
    device: Union[str, torch.device],
) -> CompositeLoss:
    loss_labels = []
    loss_modules = []
    loss_weights = []
    for loss_label, loss_conf in config.items():
        loss_labels.append(loss_label)
        loss_modules.append(create_object_from_config(loss_conf))
        loss_weights.append(loss_conf.weight)

    loss_fn = CompositeLoss(
        labels=loss_labels,
        losses=loss_modules,
        weights=loss_weights,
    ).to(torch.device(device))
    return loss_fn


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: LRSchedulerDefinition,
    max_iterations: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Create a learning rate scheduler from dictionary configuration.
    """

    lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    if config.type == SchedulerType.COSINE:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=max_iterations - config.warmup_steps,
            eta_min=config.end_value,
        )
    elif config.type == SchedulerType.LINEAR:
        start_value = optimizer.param_groups[0]["lr"]
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=1.0,
            end_factor=config.end_value / start_value,
            total_iters=config.warmup_steps,
        )
    else:
        raise RuntimeError(f'Unsupported LR scheduler "{config.type}".')

    if config.warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=1e-5,
            end_factor=1.0,
            total_iters=config.warmup_steps,
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup_scheduler, lr_scheduler],
            milestones=[config.warmup_steps],
        )

    return lr_scheduler
