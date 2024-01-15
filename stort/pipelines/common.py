from typing import Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from ..config import (
    ObjectDefinition,
    LossDefinition,
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
