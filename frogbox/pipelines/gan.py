"""
# Training a GAN

The GAN pipeline is similar to the supervised pipelines, except that it adds
another model, the discriminator, with its own loss function(s).

The discriminator model is configured in the `disc_model ` similarly to the
(generator) model:

```json
{
    "type": "gan",
    "model": {
        "class_name": "models.generator.MyGenerator",
        "params": { ... }
    },
    "disc_model": {
        "class_name": "models.disciminator.MyDiscriminator",
        "params": { ... }
    },
    ...
}
```

## Loss functions

The `GANPipeline` requires two different loss functions: `losses` defines the
loss function for the generator and `disc_losses` defines the loss function for
the disciminator.

The discriminator loss takes two optional arguments, `disc_real` and
`disc_fake`, and the generator loss takes one optional argument, `disc_fake`.
These tensors contain the predictions from the discriminator model
when passed the batch of real and fake data, respectively.

They are computed (roughly) like this:

```python
x, y = fetch_data(dataset)
y_pred = model(x)
disc_real = disc_model(y)
disc_fake = disc_model(y_pred)
```

Note: These arguments are optional keyword arguments and thus their names must
match exactly. Example:

```python
class DiscriminatorLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, input, target, disc_real, disc_fake):
        loss_real = self.loss_fn(disc_real, torch.ones_like(disc_real))
        loss_fake = self.loss_fn(disc_fake, torch.zeros_like(disc_fake))
        return loss_real + loss_fake
```

## Updating models at different intervals

It is possible to update the generator and disciminator models at different
intervals using the `update_interval` and `disc_update_interval` fields.
For instance, in order to update the discriminator only every five iterations:

```json
{
    "type": "gan",
    "update_interval": 1,
    "disc_update_interval": 5,
    ...
}
```
"""

from typing import Any, Dict, Callable, Union, Optional, Sequence
from os import PathLike
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from ignite.engine import Engine, Events
from ignite.handlers import global_step_from_engine
from ignite.contrib.handlers import ProgressBar
from accelerate import Accelerator
from .pipeline import Pipeline
from .common import (
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
from .logger import AccelerateLogger


class GANPipeline(Pipeline):
    """GAN pipeline."""

    config: GANConfig
    evaluator: Engine
    datasets: Dict[str, Dataset]
    loaders: Dict[str, DataLoader]
    model: torch.nn.Module
    disc_model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    disc_optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    disc_lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    loss_fn: CompositeLoss
    disc_loss_fn: CompositeLoss

    def __init__(
        self,
        config: GANConfig,
        checkpoint: Optional[Union[str, PathLike]] = None,
        checkpoint_keys: Optional[Sequence[str]] = None,
        checkpoint_dir: Union[str, PathLike] = Path("checkpoints"),
        logging: str = "online",
        wandb_id: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        group: Optional[str] = None,
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
        trainer_input_transform : Callable
            Function that receives tensors `x` and `y` and outputs tuple of
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
            Function that receives tensors `x` and `y` and outputs tuple of
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

        # Parse config
        self.config = config
        log_interval = parse_log_interval(config.log_interval)

        # Create accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="wandb",
        )

        # Set up trackers
        self.accelerator.init_trackers(
            config.project,
            config=self.config.model_dump(),
            init_kwargs={
                "wandb": {
                    "id": wandb_id,
                    "mode": logging,
                    "tags": tags,
                    "group": group,
                    "resume": "allow" if logging == "online" else None,
                }
            },
        )

        # Create datasets and data loaders
        self.datasets, self.loaders = self._create_data_loaders(
            batch_size=config.batch_size,
            loader_workers=config.loader_workers,
            datasets=config.datasets,
            loaders=config.loaders,
        )

        # Create models
        self.model = create_object_from_config(config.model)
        self.optimizer = create_object_from_config(
            config=config.optimizer,
            params=self.model.parameters(),
        )

        self.disc_model = create_object_from_config(config.disc_model)
        self.disc_optimizer = create_object_from_config(
            config=config.disc_optimizer,
            params=self.disc_model.parameters(),
        )

        # Create learning rate schedulers
        max_iterations = len(self.loaders["train"]) * config.max_epochs
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

        # Wrap with accelerator
        self.model, self.optimizer, self.lr_scheduler = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.lr_scheduler
            )
        )
        self.disc_model, self.disc_optimizer, self.disc_lr_scheduler = (
            self.accelerator.prepare(
                self.disc_model, self.disc_optimizer, self.disc_lr_scheduler
            )
        )
        for split in self.loaders.keys():
            self.loaders[split] = self.accelerator.prepare(self.loaders[split])

        # Create trainer
        self.loss_fn = create_composite_loss(
            config.losses, self.accelerator.device
        )
        self.disc_loss_fn = create_composite_loss(
            config.disc_losses, self.accelerator.device
        )
        self.trainer = create_gan_trainer(
            accelerator=self.accelerator,
            model=self.model,
            disc_model=self.disc_model,
            optimizer=self.optimizer,
            disc_optimizer=self.disc_optimizer,
            scheduler=self.lr_scheduler,
            disc_scheduler=self.disc_lr_scheduler,
            loss_fn=self.loss_fn,
            disc_loss_fn=self.disc_loss_fn,
            clip_grad_norm=config.clip_grad_norm,
            clip_grad_value=config.clip_grad_value,
            input_transform=trainer_input_transform,
            model_transform=trainer_model_transform,
            disc_model_transform=trainer_disc_model_transform,
            output_transform=trainer_output_transform,
        )

        # Create evaluator
        metrics = {}
        for metric_label, metric_conf in config.metrics.items():
            metrics[metric_label] = create_object_from_config(metric_conf)

        self.evaluator = create_supervised_evaluator(
            accelerator=self.accelerator,
            model=self.model,
            metrics=metrics,
            input_transform=evaluator_input_transform,
            model_transform=evaluator_model_transform,
            output_transform=evaluator_output_transform,
        )

        self.trainer.add_event_handler(
            event_name=log_interval,
            handler=lambda: self.evaluator.run(self.loaders["val"]),
        )

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
        self._setup_checkpoint(
            to_save=to_save,
            checkpoint_dir=checkpoint_dir,
            to_unwrap=["model", "disc_model"],
        )

        if checkpoint:
            self._load_checkpoint(checkpoint, checkpoint_keys)

        # Set up logging
        if self.accelerator.is_main_process:
            ProgressBar(desc="Train", ncols=80).attach(self.trainer)
            ProgressBar(desc="Val", ncols=80).attach(self.evaluator)

            def log_losses():
                fns = (self.loss_fn, self.disc_loss_fn)
                prefixes = ("loss", "disc_loss")
                for prefix, fn in zip(prefixes, fns):
                    labels = [f"{prefix}/{label}" for label in fn.labels]
                    losses = dict(zip(labels, fn.last_values))
                    self.log(losses)

            self.trainer.add_event_handler(
                Events.ITERATION_COMPLETED, log_losses
            )
            self.evaluator.add_event_handler(Events.COMPLETED, self.checkpoint)

            self.logger = AccelerateLogger(self.accelerator)
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

    def run(self) -> None:
        try:
            self.trainer.run(
                data=self.loaders["train"],
                max_epochs=self.config.max_epochs,
            )
        except KeyboardInterrupt:
            self.print("Interrupted")
        finally:
            self.accelerator.end_training()
