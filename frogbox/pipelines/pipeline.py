from typing import Callable, Union, Optional, Any, Dict, Sequence
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from ignite.engine import Engine, Events, CallableEventWithFilter
from ignite.handlers import global_step_from_engine, Checkpoint, DiskSaver
from ignite.contrib.handlers.wandb_logger import WandBLogger
import torch
import wandb
from ..config import Config, CheckpointMode, parse_log_interval


class Pipeline(ABC):
    """Pipeline abstract base class."""
    config: Config
    trainer: Engine
    device: torch.device
    logger: WandBLogger
    checkpoint: Checkpoint
    run_name: str

    @abstractmethod
    def run(self) -> None: ...

    def _setup_logger(
        self,
        wandb_id: Optional[str],
        mode: Optional[str] = "online",
        tags: Optional[Sequence[str]] = None,
        group: Optional[str] = None,
    ) -> None:
        assert mode in (
            "online",
            "offline",
        ), 'mode must be one of "online" or "offline".'

        self.logger = WandBLogger(
            id=wandb_id,
            mode=mode,
            tags=tags,
            group=group,
            resume="allow" if mode == "online" else None,
            project=self.config.project,
            config=dict(config=self.config.model_dump()),
        )

        if wandb.run is not None:
            if mode == "online":
                self.run_name = wandb.run.name
            else:
                self.run_name = f"offline-{wandb.run.id}"

    def _setup_checkpoint(
        self,
        to_save: Dict[str, Any],
        checkpoint_dir: Union[str, PathLike],
    ) -> None:
        assert self.run_name is not None, "Run name not set."

        log_interval = parse_log_interval(self.config.log_interval)
        run_dir = Path(checkpoint_dir) / self.run_name

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

        save_handler = DiskSaver(
            dirname=str(run_dir),
            create_dir=True,
            require_empty=False,
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

        # Write config.json to checkpoint directory
        config_json = self.config.model_dump_json(
            indent=True, exclude_none=True
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "config.json").open("w") as fp:
            fp.write(config_json)

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
            to_load=to_load, checkpoint=torch.load(str(path), "cpu")
        )

    def install_callback(
        self,
        event: Union[Events, CallableEventWithFilter],
        callback: Callable[["Pipeline"], None],
    ) -> None:
        self.trainer.add_event_handler(
            event_name=event,
            handler=callback,
            pipeline=self,
        )
