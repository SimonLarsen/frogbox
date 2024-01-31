from typing import Tuple, Dict
from math import ceil
import torch
from torch.utils.data import Dataset, DataLoader
from ignite.handlers import (
    ParamScheduler,
    CosineAnnealingScheduler,
    LinearCyclicalScheduler,
    create_lr_scheduler_with_warmup,
)
from ..config import (
    ObjectDefinition,
    LossDefinition,
    SchedulerType,
    LRSchedulerDefinition,
    create_object_from_config,
)
from .composite_loss import CompositeLoss


def create_data_loaders(
    batch_size: int,
    loader_workers: int,
    datasets: Dict[str, ObjectDefinition],
    loaders: Dict[str, ObjectDefinition] = dict(),
) -> Tuple[Dict[str, Dataset], Dict[str, DataLoader]]:
    out_datasets = {}
    out_loaders = {}
    for split in datasets.keys():
        ds = create_object_from_config(datasets[split])
        out_datasets[split] = ds

        if split in loaders:
            out_loaders[split] = create_object_from_config(
                loaders[split],
                dataset=ds,
                batch_size=batch_size,
                num_workers=loader_workers,
            )
        else:
            out_loaders[split] = DataLoader(
                dataset=ds,
                batch_size=batch_size,
                num_workers=loader_workers,
                shuffle=split == "train",
            )

    return out_datasets, out_loaders


def create_composite_loss(
    config: Dict[str, LossDefinition],
    device: torch.device,
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
    ).to(device)
    return loss_fn


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: LRSchedulerDefinition,
    max_iterations: int,
) -> ParamScheduler:
    """
    Create a learning rate scheduler from dictionary configuration.
    """
    cycle_size = ceil(max_iterations / config.cycles)

    lr_scheduler: ParamScheduler
    if config.type == SchedulerType.COSINE:
        lr_scheduler = CosineAnnealingScheduler(
            optimizer=optimizer,
            param_name="lr",
            start_value=config.start_value,
            end_value=config.end_value,
            cycle_size=cycle_size,
            start_value_mult=config.start_value_mult,
            end_value_mult=config.end_value_mult,
        )
    elif config.type == SchedulerType.LINEAR:
        lr_scheduler = LinearCyclicalScheduler(
            optimizer=optimizer,
            param_name="lr",
            start_value=config.start_value,
            end_value=config.end_value,
            cycle_size=cycle_size,
            start_value_mult=config.start_value_mult,
            end_value_mult=config.end_value_mult,
        )
    else:
        raise RuntimeError(f'Unsupported LR scheduler "{config.type}".')

    if config.warmup_steps > 0:
        lr_scheduler = create_lr_scheduler_with_warmup(
            lr_scheduler=lr_scheduler,
            warmup_start_value=0.0,
            warmup_end_value=config.start_value,
            warmup_duration=config.warmup_steps,
        )

    return lr_scheduler
