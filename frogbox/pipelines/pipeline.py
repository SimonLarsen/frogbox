from typing import (
    Dict,
    Any,
    Optional,
    Tuple,
    Mapping,
    Sequence,
    Callable,
)
from os import PathLike
from abc import ABC, abstractmethod
import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from ..engines.engine import Engine
from ..engines.events import MatchableEvent
from ..config import (
    Config,
    ObjectDefinition,
    LossDefinition,
    parse_log_interval,
    create_object_from_config,
)
from ..handlers.checkpoint import Checkpoint
from .composite_loss import CompositeLoss
from .name_generation import generate_name


class Pipeline(ABC):
    """Pipeline abstract base class."""

    config: Config
    accelerator: Accelerator
    trainer: Engine

    _run_name: Optional[str] = None

    def _create_data_loaders(
        self,
        batch_size: int,
        loader_workers: int,
        datasets: Mapping[str, ObjectDefinition],
        loaders: Optional[Mapping[str, ObjectDefinition]] = None,
    ) -> Tuple[Dict[str, Dataset], Dict[str, DataLoader]]:
        if loaders is None:
            loaders = {}

        out_datasets = {}
        out_loaders = {}

        with self.accelerator.local_main_process_first():
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

    def _create_composite_loss(
        self,
        config: Mapping[str, LossDefinition],
        transforms: Optional[Mapping[str, Callable[[Any, Any], Any]]] = None,
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
            transforms=transforms,
        )
        loss_fn = loss_fn.to(self.device)
        return loss_fn

    def _setup_tracking(
        self,
        mode: str,
        wandb_id: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        group: Optional[str] = None,
    ) -> None:
        self.accelerator.init_trackers(
            project_name=self.config.project,
            config=self.config.model_dump(),
            init_kwargs={
                "wandb": {
                    "id": wandb_id,
                    "mode": mode,
                    "name": self.run_name,
                    "tags": tags,
                    "group": group,
                }
            },
        )

    def _load_checkpoint(
        self,
        path: str | PathLike,
        to_load: Mapping[str, Any],
        to_unwrap: Optional[Sequence[str]] = None,
        keys: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Load checkpoint from file.

        Attributes
        ----------
        path : path-like
            Path to checkpoint file.
        to_load : mapping
            Mapping with objects to load.
        to_unwrap : list of str
            Keys for objects to unwrap before loading.
        keys : list of str (optional)
            List of keys to filter.
        """
        if keys is None:
            keys = list(to_load.keys())
        to_load = {k: to_load[k] for k in keys}

        Checkpoint.load_checkpoint(
            accelerator=self.accelerator,
            path=path,
            to_load=to_load,
            to_unwrap=to_unwrap,
        )
        self.accelerator.wait_for_everyone()

    def install_callback(
        self,
        event: MatchableEvent,
        callback: Callable[["Pipeline"], None],
        engine: str = "trainer",
        only_main_process: bool = False,
        **kwargs,
    ) -> None:
        """
        Install callback in pipeline.

        Parameters
        ----------
        event : MatchableEvent
            Event to trigger callback.
        callback : callable
            Callback function.
            Should take a single argument `pipeline` and return nothing.
        engine : str
            Which engine to install callback in. Defaults to "trainer".
        only_main_process : bool
            Install only in main process. Only affects distributed setups.
        kwargs : keyword arguments
            Optional keyword arguments to be passed to callback.
        """
        if not only_main_process or self.accelerator.is_main_process:
            target = getattr(self, engine)
            if not isinstance(target, Engine):
                raise ValueError(
                    f"'{engine}' is not an engine. Cannot install callback."
                )

            target.add_event_handler(event, callback, self, **kwargs)

    @abstractmethod
    def run(self) -> None: ...

    def log(self, data: Dict[str, Any]) -> None:
        """Log data to tracker(s)."""
        self.accelerator.log(data, step=self.trainer.iteration)

    def print(self, *args, **kwargs) -> None:
        """Drop in replacement of `print()` to only print once per server."""
        self.accelerator.print(*args, **kwargs)

    def gather_for_metrics(self, input_data, use_gather_object: bool = False):
        """
        Gathers `input_data` and potentially drops duplicates in the last
        batch if on a distributed system. Should be used for gathering the
        inputs and targets for metric calculation.

        Wrapper around `Accelerator.gather_for_metrics()
        <https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator.gather_for_metrics>`_.
        """  # noqa: E501, W505
        return self.accelerator.gather_for_metrics(
            input_data, use_gather_object
        )

    @property
    def device(self) -> torch.device:
        return self.accelerator.device

    @property
    def is_main_process(self) -> bool:
        """True for one process only."""
        return self.accelerator.is_main_process

    @property
    def is_local_main_process(self) -> bool:
        """True for one process per server."""
        return self.accelerator.is_local_main_process

    @property
    def run_name(self) -> str:
        """Get name of current run."""
        if self._run_name is None:
            suffix = generate_name()
            now = datetime.datetime.now(datetime.timezone.utc)
            timestamp = now.strftime("%Y%m%d-%H%M")
            self._run_name = f"{timestamp}-{suffix}"
        return self._run_name

    @property
    def log_interval(self) -> MatchableEvent:
        return parse_log_interval(self.config.log_interval)
