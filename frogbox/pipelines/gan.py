from typing import Any, Dict, Callable, Union, Optional, Sequence
from os import PathLike
from pathlib import Path
from math import ceil
import torch
from torch.utils.data import Dataset, DataLoader
from ignite.engine import Engine, Events, _prepare_batch
from ignite.handlers import ParamScheduler, global_step_from_engine
from ignite.contrib.handlers import ProgressBar
import wandb
from .pipeline import Pipeline
from .common import (
    create_data_loaders,
    create_composite_loss,
    create_lr_scheduler,
)
from ..config import (
    GANConfig,
    create_object_from_config,
    parse_log_interval,
)
from ..engines.supervised import create_supervised_evaluator
from ..engines.gan import create_gan_trainer
from .composite_loss import CompositeLoss


class GANPipeline(Pipeline):
    config: GANConfig
    evaluator: Engine
    device: torch.device
    datasets: Dict[str, Dataset]
    loaders: Dict[str, DataLoader]
    model: torch.nn.Module
    disc_model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    disc_optimizer: torch.optim.Optimizer
    lr_scheduler: ParamScheduler
    disc_lr_scheduler: ParamScheduler
    loss_fn: CompositeLoss
    disc_loss_fn: CompositeLoss

    def __init__(
        self,
        config: GANConfig,
        device: Union[str, torch.device],
        checkpoint: Optional[Union[str, PathLike]] = None,
        checkpoint_keys: Optional[Sequence[str]] = None,
        checkpoint_dir: Union[str, PathLike] = Path("checkpoints"),
        logging: str = "online",
        wandb_id: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        group: Optional[str] = None,
        prepare_batch: Callable = _prepare_batch,
        trainer_input_transform: Callable[[Any, Any], Any] = lambda x, y: (
            x,
            y,
        ),
        trainer_model_transform: Callable[[Any], Any] = lambda output: output,
        trainer_disc_model_transform: Callable[
            [Any], Any
        ] = lambda output: output,
        trainer_output_transform: Callable[
            [Any, Any, Any, torch.Tensor, torch.Tensor], Any
        ] = lambda x, y, y_pred, loss, disc_loss: (
            loss.item(),
            disc_loss.item(),
        ),
        evaluator_input_transform: Callable[[Any, Any], Any] = lambda x, y: (
            x,
            y,
        ),
        evaluator_model_transform: Callable[
            [Any], Any
        ] = lambda output: output,
        evaluator_output_transform: Callable[
            [Any, Any, Any], Any
        ] = lambda x, y, y_pred: (y_pred, y),
    ):
        """
        Create GAN pipeline.

        Parameters
        ----------
        config : GANConfig
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
        wandb_id : str
            W&B run ID to resume from.
        tags : list of str
            List of tags to add to the run in W&B.
        group : str
            Group to add run to in W&B.
        prepare_batch : Callable
            Function that receives `batch`, `device`, `non_blocking` and
            outputs tuple of tensors `(batch_x, batch_y)`.
        trainer_input_transform : Callable
            Function that receives tensors `y` and `y` and outputs tuple of
            tensors `(x, y)`.
        trainer_model_transform : Callable
            Function that receives the output from the model during training
            and converts it into the form as required by the loss function.
        trainer_disc_model_transform : Callable
            Function that receives the output from the discriminator
            during training and converts it into the form as required
            by the loss function.
        trainer_output_transform : Callable
            Function that receives `x`, `y`, `y_pred`, `loss` and returns value
            to be assigned to trainer's `state.output` after each iteration.
            Default is returning `loss.item()`.
        evaluator_input_transform : Callable
            Function that receives tensors `y` and `y` and outputs tuple of
            tensors `(x, y)`.
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
        assert (
            config.gradient_accumulation_steps == 1
        ), "Gradient accumulation not supported for GAN pipeline."

        self.device = torch.device(device)

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

        # Create models
        self.model = create_object_from_config(config.model).to(device)
        self.optimizer = create_object_from_config(
            config=config.optimizer,
            params=self.model.parameters(),
        )

        self.disc_model = create_object_from_config(config.disc_model).to(
            device
        )
        self.disc_optimizer = create_object_from_config(
            config=config.disc_optimizer,
            params=self.disc_model.parameters(),
        )

        # Create trainer
        self.loss_fn = create_composite_loss(config.losses, device)
        self.disc_loss_fn = create_composite_loss(config.disc_losses, device)
        self.trainer = create_gan_trainer(
            model=self.model,
            disc_model=self.disc_model,
            optimizer=self.optimizer,
            disc_optimizer=self.disc_optimizer,
            loss_fn=self.loss_fn,
            disc_loss_fn=self.disc_loss_fn,
            device=device,
            amp=config.amp,
            clip_grad_norm=config.clip_grad_norm,
            prepare_batch=prepare_batch,
            input_transform=trainer_input_transform,
            model_transform=trainer_model_transform,
            disc_model_transform=trainer_disc_model_transform,
            output_transform=trainer_output_transform,
        )
        ProgressBar(desc="Train", ncols=80).attach(self.trainer)

        # Create learning rate schedulers
        max_iterations = ceil(
            len(self.datasets["train"])  # type: ignore[arg-type]
            / config.batch_size
            * config.max_epochs
        )
        self.lr_scheduler = create_lr_scheduler(
            optimizer=self.optimizer,
            config=config.lr_scheduler,
            max_iterations=max_iterations,
        )
        self.disc_lr_scheduler = create_lr_scheduler(
            optimizer=self.disc_optimizer,
            config=config.disc_lr_scheduler,
            max_iterations=max_iterations,
        )
        self.trainer.add_event_handler(
            Events.ITERATION_STARTED, self.lr_scheduler
        )
        self.trainer.add_event_handler(
            Events.ITERATION_STARTED, self.disc_lr_scheduler
        )

        # Create evaluator
        metrics = {}
        for metric_label, metric_conf in config.metrics.items():
            metrics[metric_label] = create_object_from_config(metric_conf)

        self.evaluator = create_supervised_evaluator(
            model=self.model,
            metrics=metrics,
            device=device,
            amp=config.amp,
            prepare_batch=prepare_batch,
            input_transform=evaluator_input_transform,
            model_transform=evaluator_model_transform,
            output_transform=evaluator_output_transform,
        )
        ProgressBar(desc="Val", ncols=80).attach(self.evaluator)

        self.trainer.add_event_handler(
            event_name=log_interval,
            handler=lambda: self.evaluator.run(self.loaders["val"]),
        )

        # Set up logging
        self._setup_logger(wandb_id, logging, tags, group)
        self.logger.attach_output_handler(
            engine=self.trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag="train",
            output_transform=lambda losses: {
                "loss": losses[0],
                "disc_loss": losses[1],
            },
        )
        self.logger.attach_opt_params_handler(
            engine=self.trainer,
            event_name=Events.ITERATION_COMPLETED,
            optimizer=self.optimizer,
            tag="optimizer",
            param_name="lr",
        )
        self.logger.attach_opt_params_handler(
            engine=self.trainer,
            event_name=Events.ITERATION_COMPLETED,
            optimizer=self.disc_optimizer,
            tag="disc_optimizer",
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
            fns = (self.loss_fn, self.disc_loss_fn)
            prefixes = ("loss", "disc_loss")
            for prefix, fn in zip(prefixes, fns):
                labels = [f"{prefix}/{label}" for label in fn.labels]
                losses = dict(zip(labels, fn.last_values))
                wandb.log(step=trainer.state.iteration, data=losses)

        # Set up checkpoints
        to_save = {
            "model": self.model,
            "disc_model": self.disc_model,
            "optimizer": self.optimizer,
            "disc_optimizer": self.disc_optimizer,
            "trainer": self.trainer,
            "lr_scheduler": self.lr_scheduler,
            "disc_lr_scheduler": self.disc_lr_scheduler,
        }

        self._setup_checkpoint(to_save, checkpoint_dir)
        self.evaluator.add_event_handler(Events.COMPLETED, self.checkpoint)

        if checkpoint:
            self._load_checkpoint(checkpoint, checkpoint_keys)

    def run(self) -> None:
        self.trainer.run(
            data=self.loaders["train"],
            max_epochs=self.config.max_epochs,
        )
