from typing import (
    Callable,
    Union,
    Optional,
    Any,
    Dict,
    Sequence,
    Mapping,
    Tuple,
)
from abc import ABC, abstractmethod
import os
from os import PathLike
from pathlib import Path
import datetime
import tempfile
import stat
import torch
from torch.utils.data import Dataset, DataLoader
from ignite.engine import Engine, Events, CallableEventWithFilter
from ignite.handlers import global_step_from_engine, Checkpoint
from ignite.handlers.checkpoint import BaseSaveHandler
from ignite.handlers.base_logger import BaseLogger
from accelerate import Accelerator
from ..config import (
    Config,
    CheckpointMode,
    ObjectDefinition,
    create_object_from_config,
    parse_log_interval,
)


class NoneSaveHandler(BaseSaveHandler):
    """@private"""
    def __call__(
        self,
        checkpoint: Mapping,
        filename: str,
        metadata: Optional[Mapping] = None,
    ) -> None:
        pass

    def remove(self, filename: str) -> None:
        pass


class AccelerateDiskSaver(BaseSaveHandler):
    """@private"""
    def __init__(
        self,
        dirname: Union[str, PathLike],
        accelerator: Accelerator,
        to_unwrap: Optional[Sequence[str]] = None,
        atomic: bool = True,
        **kwargs,
    ):
        self.dirname = Path(dirname).expanduser()
        self.accelerator = accelerator
        self.to_unwrap = to_unwrap
        self.atomic = atomic
        self.kwargs = kwargs

        if not self.dirname.exists():
            self.dirname.mkdir(parents=True)

    def __call__(
        self,
        checkpoint: Mapping,
        filename: str,
        metadata: Optional[Mapping] = None,
    ) -> None:
        to_unwrap = self.to_unwrap if self.to_unwrap else []
        unwrapped_checkpoint = {}
        for key in checkpoint:
            unwrapped_checkpoint[key] = (
                self.accelerator.unwrap_model(checkpoint[key])
                if key in to_unwrap
                else checkpoint[key]
            )

        path = self.dirname / filename
        self._save_func(unwrapped_checkpoint, path, self.accelerator.save)

    def _save_func(
        self, checkpoint: Mapping, path: Path, func: Callable
    ) -> None:
        if not self.atomic:
            func(checkpoint, path, **self.kwargs)
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, dir=self.dirname)
            tmp_file = tmp.file
            tmp_name = tmp.name
            try:
                func(checkpoint, tmp_file, **self.kwargs)
            except BaseException:
                tmp.close()
                os.remove(tmp_name)
                raise
            else:
                tmp.close()
                os.replace(tmp.name, path)
                # append group/others read mode
                os.chmod(
                    path, os.stat(path).st_mode | stat.S_IRGRP | stat.S_IROTH
                )

    def remove(self, filename: str) -> None:
        path = self.dirname / filename
        path.unlink()


class Pipeline(ABC):
    """Pipeline abstract base class."""

    config: Config
    accelerator: Accelerator
    trainer: Engine
    logger: BaseLogger
    checkpoint: Checkpoint

    @abstractmethod
    def run(self) -> None: ...

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

    def _setup_checkpoint(
        self,
        to_save: Dict[str, Any],
        checkpoint_dir: Union[str, PathLike],
        to_unwrap: Optional[Sequence[str]] = None,
    ) -> None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
        run_name = f"{self.config.project}_{timestamp}"

        log_interval = parse_log_interval(self.config.log_interval)
        run_dir = Path(checkpoint_dir) / run_name

        save_handler: BaseSaveHandler
        if self.accelerator.is_main_process:
            save_handler = AccelerateDiskSaver(
                dirname=str(run_dir),
                accelerator=self.accelerator,
                to_unwrap=to_unwrap,
            )

            config_json = self.config.model_dump_json(
                indent=True, exclude_none=True
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            with (run_dir / "config.json").open("w") as fp:
                fp.write(config_json)
        else:
            save_handler = NoneSaveHandler()

        score_function = None
        if self.config.checkpoint_metric:
            score_function = Checkpoint.get_default_score_fn(
                metric_name=self.config.checkpoint_metric,
                score_sign=(
                    1.0
                    if self.config.checkpoint_mode == CheckpointMode.MAX
                    else -1.0
                ),
            )

        self.checkpoint = Checkpoint(
            to_save=to_save,
            save_handler=save_handler,
            score_name=self.config.checkpoint_metric,
            score_function=score_function,
            n_saved=self.config.checkpoint_n_saved,
            global_step_transform=global_step_from_engine(
                self.trainer,
                Events(log_interval.value),
            ),
        )

    def _load_checkpoint(
        self,
        path: Union[str, PathLike],
        keys: Optional[Sequence[str]] = None,
    ) -> None:
        to_save = self.checkpoint.to_save
        if keys is None:
            keys = list(to_save.keys())
        to_load = {k: to_save[k] for k in keys}

        Checkpoint.load_objects(
            to_load=to_load,
            checkpoint=torch.load(
                str(path), map_location="cpu", weights_only=True
            ),
        )

    def install_callback(
        self,
        event: Union[Events, CallableEventWithFilter],
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
        """Gathers `input_data` and potentially drops duplicates in the last
        batch if on a distributed system. Should be used for gathering the
        inputs and targets for metric calculation.

        Wrapper around `Accelerator.gather_for_metrics()
        <https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator.gather_for_metrics>`_."""  # noqa: E501, W505
        return self.accelerator.gather_for_metrics(
            input_data, use_gather_object
        )
