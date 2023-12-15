from typing import Dict, Optional, Union, Sequence
from os import PathLike
from math import ceil
import torch
from torch.utils.data import Dataset, DataLoader
from ignite.engine import Engine, Events
from ignite.handlers import global_step_from_engine, Checkpoint, TerminateOnNan
from ignite.contrib.handlers import ProgressBar, WandBLogger
import wandb
from .pipeline import Pipeline
from .common import create_data_loaders, create_composite_loss
from ..config import (
    SupervisedConfig,
    CheckpointMode,
    create_object_from_config,
    parse_log_interval,
    create_lr_scheduler_from_config,
)
from ..engines.supervised import (
    create_supervised_trainer,
    create_supervised_evaluator,
)


class SupervisedPipeline(Pipeline):
    config: SupervisedConfig
    evaluator: Engine
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    datasets: Dict[str, Dataset]
    loaders: Dict[str, DataLoader]

    def __init__(
        self,
        config: SupervisedConfig,
        device: torch.device,
        checkpoint: Optional[Union[str, PathLike]] = None,
        checkpoint_keys: Optional[Sequence[str]] = None,
    ):
        self.device = device

        # Parse config
        self.config = config
        log_interval = parse_log_interval(config.log_interval)

        # Create datasets and data loaders
        self.datasets, self.loaders = create_data_loaders(
            batch_size=config.batch_size,
            loader_workers=config.loader_workers,
            datasets=config.datasets,
            loaders=config.loaders,
        )

        # Create model
        self.model = create_object_from_config(config.model).to(device)
        self.optimizer = create_object_from_config(
            config=config.optimizer,
            params=self.model.parameters(),
        )

        # Create trainer
        loss_fn = create_composite_loss(config.losses, device)
        self.trainer = create_supervised_trainer(
            config=config,
            model=self.model,
            optimizer=self.optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        self.trainer.add_event_handler(
            Events.ITERATION_COMPLETED, TerminateOnNan()
        )
        ProgressBar(desc="Train", ncols=80).attach(self.trainer)

        # Create learning rate scheduler
        max_iterations = ceil(
            len(self.datasets["train"])  # type: ignore[arg-type]
            / config.batch_size
            * config.max_epochs
        )
        lr_scheduler = create_lr_scheduler_from_config(
            optimizer=self.optimizer,
            config=config.lr_scheduler,
            max_iterations=max_iterations,
        )
        self.trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)

        # Create evaluator
        metrics = {}
        for metric_label, metric_conf in config.metrics.items():
            metrics[metric_label] = create_object_from_config(metric_conf)

        self.evaluator = create_supervised_evaluator(
            config=config,
            model=self.model,
            metrics=metrics,
            device=device,
        )
        ProgressBar(desc="Val", ncols=80).attach(self.evaluator)

        @self.trainer.on(log_interval)
        def _log_validation():
            self.evaluator.run(self.loaders["val"])

        # Set up logging
        wandb_logger = WandBLogger(
            resume="allow",
            project=config.project,
            config=dict(config=config.model_dump()),
        )
        self._wandb_id = wandb.run.id  # type: ignore[union-attr]
        self._run_name = wandb.run.name  # type: ignore[union-attr]

        wandb_logger.attach_output_handler(
            engine=self.trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag="train",
            output_transform=lambda loss: {"loss": loss},
        )

        wandb_logger.attach_opt_params_handler(
            engine=self.trainer,
            event_name=Events.ITERATION_COMPLETED,
            optimizer=self.optimizer,
            param_name="lr",
        )

        wandb_logger.attach_output_handler(
            engine=self.evaluator,
            event_name=Events.COMPLETED,
            tag="val",
            metric_names="all",
            global_step_transform=global_step_from_engine(
                self.trainer, Events.ITERATION_COMPLETED
            ),
        )

        @self.trainer.on(Events.ITERATION_COMPLETED)
        def log_losses(trainer):
            labels = ["loss/" + label for label in loss_fn.labels]
            losses = dict(zip(labels, loss_fn.last_values))
            wandb.log(step=trainer.state.iteration, data=losses)

        # Set up checkpoints
        to_save = {
            "model": self.model,
            "optimizer": self.optimizer,
            "trainer": self.trainer,
            "lr_scheduler": lr_scheduler,
        }

        score_function = None
        if config.checkpoint_metric:
            score_function = Checkpoint.get_default_score_fn(
                metric_name=config.checkpoint_metric,
                score_sign=(
                    1.0
                    if config.checkpoint_mode == CheckpointMode.MAX
                    else -1.0
                ),
            )

        checkpoint_handler = Checkpoint(
            to_save=to_save,
            save_handler=f"checkpoints/{self._run_name}",
            filename_prefix="best",
            score_name=config.checkpoint_metric,
            score_function=score_function,
            n_saved=config.checkpoint_n_saved,
            global_step_transform=global_step_from_engine(self.trainer),
        )
        self.evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler)

        # Load checkpoint
        if checkpoint:
            if not checkpoint_keys:
                checkpoint_keys = list(to_save.keys())
            to_load = {k: to_save[k] for k in checkpoint_keys}

            Checkpoint.load_objects(
                to_load=to_load,
                checkpoint=torch.load(str(checkpoint), "cpu"),
            )

    def run(self) -> None:
        self.trainer.run(
            data=self.loaders["train"],
            max_epochs=self.config.max_epochs,
        )
