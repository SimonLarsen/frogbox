from typing import Tuple, Dict, Sequence
import torch
from torch.utils.data import Dataset, DataLoader
from ignite.engine import Engine
from ..callbacks import Callback, CallbackState
from ..config import Config, create_object_from_config
from .composite_loss import CompositeLoss


def create_data_loaders(
    config: Config,
) -> Tuple[Dict[str, Dataset], Dict[str, DataLoader]]:
    datasets = {}
    loaders = {}
    for split in config.datasets.keys():
        ds = create_object_from_config(config.datasets[split])
        datasets[split] = ds

        if split in config.loaders:
            loaders[split] = create_object_from_config(
                config.loaders[split],
                dataset=ds,
                batch_size=config.batch_size,
                num_workers=config.loader_workers,
            )
        else:
            loaders[split] = DataLoader(
                dataset=ds,
                batch_size=config.batch_size,
                num_workers=config.loader_workers,
                shuffle=split == "train",
            )

    return datasets, loaders


def create_composite_loss(
    config: Config,
    device: torch.device,
) -> CompositeLoss:
    loss_labels = []
    loss_modules = []
    loss_weights = []
    for loss_label, loss_conf in config.losses.items():
        loss_labels.append(loss_label)
        loss_modules.append(create_object_from_config(loss_conf))
        loss_weights.append(loss_conf.weight)

    loss_fn = CompositeLoss(
        labels=loss_labels,
        losses=loss_modules,
        weights=loss_weights,
    ).to(device)
    return loss_fn


def install_callbacks(
    trainer: Engine,
    callbacks: Sequence[Callback],
    state: CallbackState,
):
    for callback in callbacks:

        def __callback_handler():
            callback.function(state)

        trainer.add_event_handler(callback.event, __callback_handler)
