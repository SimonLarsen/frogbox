from typing import Any, Dict, Optional, Union, Sequence, Callable
from os import PathLike
from pathlib import Path
from math import ceil
import torch
from torch.utils.data import Dataset, DataLoader
from ignite.engine import Engine, Events, _prepare_batch
from ignite.handlers import global_step_from_engine
from ignite.contrib.handlers import ProgressBar
import wandb
from .pipeline import Pipeline
from .common import (
    create_data_loaders,
    create_composite_loss,
    create_lr_scheduler,
)
from ..config import (
    SupervisedConfig,
    create_object_from_config,
    parse_log_interval,
)
from ..engines.supervised import (
    create_supervised_trainer,
    create_supervised_evaluator,
)
from .composite_loss import CompositeLoss


class SupervisedPipeline(Pipeline):
    config: SupervisedConfig
    evaluator: Engine
    datasets: Dict[str, Dataset]
    loaders: Dict[str, DataLoader]
    model: torch.nn.Module
    loss_fn: CompositeLoss
    optimizer: torch.optim.Optimizer

    def __init__(
        self,
        config: SupervisedConfig,
        device: torch.device,
        checkpoint: Optional[Union[str, PathLike]] = None,
        checkpoint_keys: Optional[Sequence[str]] = None,
        checkpoint_dir: Union[str, PathLike] = Path("checkpoints"),
        logging: str = "online",
        wandb_id: Optional[str] = None,
        prepare_batch: Callable = _prepare_batch,
        trainer_model_transform: Callable[[Any], Any] = lambda output: output,
        trainer_output_transform: Callable[
            [Any, Any, Any, torch.Tensor], Any
        ] = lambda x, y, y_pred, loss: loss.item(),
        evaluator_model_transform: Callable[
            [Any], Any
        ] = lambda output: output,
        evaluator_output_transform: Callable[
            [Any, Any, Any], Any
        ] = lambda x, y, y_pred: (y_pred, y),
    ):
        """
        Create supervised pipeline.

        Parameters
        ----------
        config : SupervisedConfig
            Pipeline configuration.
        device : torch.device
            CUDA device. Can be CPU or GPU. Model will not be moved.
        checkpoint : path-like
            Path to experiment checkpoint.
        checkpoint_keys : list of str
            List of keys for objects to load from checkpoint.
            Defaults to all keys.
        checkpoint_dir : str or path
            Path to directory to store checkpoints.
        logging : str
            Logging mode. Must be either "online" or "offline".
        prepare_batch : Callable
            Function that receives `batch`, `device`, `non_blocking` and
            outputs tuple of tensors `(batch_x, batch_y)`.
        trainer_model_transform : Callable
            Function that receives the output from the model during training
            and converts it into the form as required by the loss function.
        trainer_output_transform : Callable
            Function that receives `x`, `y`, `y_pred`, `loss` and returns value
            to be assigned to trainer's `state.output` after each iteration.
            Default is returning `loss.item()`.
        evaluator_model_transform : Callable
            Function that receives the output from the model during evaluation
            and converts it into the predictions:
            `y_pred = model_transform(model(x))`.
        evaluator_output_transform : Callable
            Function that receives `x`, `y`, `y_pred` and returns value to be
            assigned to evaluator's `state.output` after each iteration.
            Default is returning `(y_pred, y)` which fits output expected by
            metrics.
        """
        logging = logging.lower()
        assert logging in ("online", "offline")

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
        self.loss_fn = create_composite_loss(config.losses, device)
        self.trainer = create_supervised_trainer(
            config=config,
            model=self.model,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            device=device,
            prepare_batch=prepare_batch,
            model_transform=trainer_model_transform,
            output_transform=trainer_output_transform,
        )
        ProgressBar(desc="Train", ncols=80).attach(self.trainer)

        # Create learning rate scheduler
        max_iterations = ceil(
            len(self.datasets["train"])  # type: ignore[arg-type]
            / config.batch_size
            * config.max_epochs
        )
        lr_scheduler = create_lr_scheduler(
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
            prepare_batch=prepare_batch,
            model_transform=evaluator_model_transform,
            output_transform=evaluator_output_transform,
        )
        ProgressBar(desc="Val", ncols=80).attach(self.evaluator)

        self.trainer.add_event_handler(
            event_name=log_interval,
            handler=lambda: self.evaluator.run(self.loaders["val"]),
        )

        # Set up logging
        self._setup_logger(wandb_id, logging)
        self.logger.attach_output_handler(
            engine=self.trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag="train",
            output_transform=lambda loss: {"loss": loss},
        )
        self.logger.attach_opt_params_handler(
            engine=self.trainer,
            event_name=Events.ITERATION_COMPLETED,
            optimizer=self.optimizer,
            param_name="lr",
        )
        self.logger.attach_output_handler(
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
            labels = ["loss/" + label for label in self.loss_fn.labels]
            losses = dict(zip(labels, self.loss_fn.last_values))
            wandb.log(step=trainer.state.iteration, data=losses)

        # Set up checkpoints
        to_save = {
            "model": self.model,
            "optimizer": self.optimizer,
            "trainer": self.trainer,
            "lr_scheduler": lr_scheduler,
        }

        self._setup_checkpoint(to_save, checkpoint_dir)
        self.evaluator.add_event_handler(Events.COMPLETED, self.checkpoint)

        # Load checkpoint
        if checkpoint:
            self._load_checkpoint(checkpoint, checkpoint_keys)

    def run(self) -> None:
        self.trainer.run(
            data=self.loaders["train"],
            max_epochs=self.config.max_epochs,
        )
