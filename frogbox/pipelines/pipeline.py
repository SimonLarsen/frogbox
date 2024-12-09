from typing import (
    Union,
    Optional,
    Any,
    Callable,
    Sequence,
    Tuple,
    Dict,
)
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
import datetime
from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader
from ignite.engine import Engine, Events, CallableEventWithFilter
from ignite.handlers import global_step_from_engine, Checkpoint
from ignite.handlers.checkpoint import BaseSaveHandler
from ignite.handlers.base_logger import BaseLogger
from accelerate import Accelerator
from accelerate.utils.other import is_compiled_module
from .save_handler import NoneSaveHandler, AccelerateDiskSaver
from .name_generation import generate_name
from ..config import (
    Config,
    CheckpointMode,
    ObjectDefinition,
    create_object_from_config,
    parse_log_interval,
)


class Pipeline(ABC):
    """Pipeline abstract base class."""

    config: Config
    accelerator: Accelerator
    trainer: Engine
    evaluator: Engine
    logger: BaseLogger
    run_name: str

    @abstractmethod
    def run(self) -> None: ...

    def _generate_name(self) -> None:
        suffix = generate_name()
        now = datetime.datetime.now(datetime.timezone.utc)
        timestamp = now.strftime("%Y%m%d-%H%M")
        self.run_name = f"{timestamp}-{suffix}"

    def _create_data_loaders(
        self,
        batch_size: int,
        loader_workers: int,
        datasets: Dict[str, ObjectDefinition],
        loaders: Optional[Dict[str, ObjectDefinition]] = None,
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

    def _setup_checkpoints(
        self,
        to_save: Dict[str, Any],
        checkpoint_dir: Union[str, PathLike],
        to_unwrap: Optional[Sequence[str]] = None,
    ) -> None:
        run_dir = Path(checkpoint_dir) / self.run_name

        save_handler: BaseSaveHandler
        if self.accelerator.is_main_process:
            save_handler = AccelerateDiskSaver(
                dirname=str(run_dir),
                accelerator=self.accelerator,
            )

            config_json = self.config.model_dump_json(
                indent=True, exclude_none=True
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            with (run_dir / "config.json").open("w") as fp:
                fp.write(config_json)
        else:
            save_handler = NoneSaveHandler()

        def evaluator_score_fn(
            engine: Engine,
            evaluator: Engine,
            metric: str,
            sign: float,
        ):
            return sign * evaluator.state.metrics[metric]

        def unwrap_to_save_fn():
            output = {}
            for k, v in to_save.items():
                if k in to_unwrap:
                    v = self.accelerator.unwrap_model(v)
                    if is_compiled_module(v):
                        v = v._orig_mod
                output[k] = v
            return output

        def handler_fn(engine: Engine, handler: Checkpoint):
            handler.to_save = unwrap_to_save_fn()
            return handler(engine)

        for checkpoint in self.config.checkpoints:
            score_function = None
            if checkpoint.metric:
                score_sign = (
                    1.0 if checkpoint.mode == CheckpointMode.MAX else -1.0
                )
                score_function = partial(
                    evaluator_score_fn,
                    evaluator=self.evaluator,
                    metric=checkpoint.metric,
                    sign=score_sign,
                )

            log_interval = parse_log_interval(checkpoint.interval)
            handler = Checkpoint(
                to_save=unwrap_to_save_fn(),
                save_handler=save_handler,
                score_name=checkpoint.metric,
                score_function=score_function,
                n_saved=checkpoint.n_saved,
                global_step_transform=global_step_from_engine(
                    self.trainer,
                    Events(log_interval.value),
                ),
            )

            self.trainer.add_event_handler(
                log_interval, partial(handler_fn, handler=handler)
            )

    def _load_checkpoint(
        self,
        path: Union[str, PathLike],
        to_load: Dict[str, Any],
        keys: Optional[Sequence[str]] = None,
    ) -> None:
        if keys is None:
            keys = list(to_load.keys())
        to_load = {k: to_load[k] for k in keys if k in to_load}

        Checkpoint.load_objects(
            to_load=to_load,
            checkpoint=torch.load(
                str(path), map_location="cpu", weights_only=True
            ),
        )

        self.accelerator.wait_for_everyone()

    def _setup_tracking(
        self,
        mode: str,
        wandb_id: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        group: Optional[str] = None,
    ) -> None:
        self.accelerator.init_trackers(
            self.config.project,
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

    def install_callback(
        self,
        event: CallableEventWithFilter,
        callback: Callable[["Pipeline"], None],
        only_main_process: bool = False,
    ) -> None:
        """
        Install callback in pipeline.

        Parameters
        ----------
        event : Events
            Event to trigger callback at.
        callback : callable
            Callback to install.
        only_main_process : bool
            Install only in main process. Only affects distributed setups.
        """
        if not only_main_process or self.accelerator.is_main_process:
            self.trainer.add_event_handler(
                event_name=event,
                handler=callback,
                pipeline=self,
            )

    def log(self, data: Dict[str, Any]) -> None:
        """Log data to tracker(s)."""
        self.accelerator.log(data, step=self.trainer.state.iteration)

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
