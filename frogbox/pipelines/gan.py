"""
# Training a GAN

The GAN pipeline is similar to the supervised pipelines, except that it adds
another model, the discriminator, with its own loss function(s).

The discriminator model is configured in the `disc_model` field similarly
to the (generator) model:

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

The discriminator loss takes two keyword arguments, `disc_real` and
`disc_fake`.
The generator loss takes one optional argument, `disc_fake`.
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
"""

from typing import Dict, Optional, Sequence, Callable, Any, Mapping
from os import PathLike
import warnings
from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from torchmetrics import Metric
from .pipeline import Pipeline
from ..config import GANConfig, create_object_from_config, parse_log_interval
from ..engines.gan import GANTrainer
from ..engines.supervised import SupervisedEvaluator
from .lr_scheduler import create_lr_scheduler
from .composite_loss import CompositeLoss
from ..handlers.output_logger import OutputLogger
from ..handlers.metric_logger import MetricLogger
from ..handlers.optimizer_logger import OptimizerLogger
from ..handlers.composite_loss_logger import CompositeLossLogger
from ..handlers.checkpoint import Checkpoint


class GANPipeline(Pipeline):
    """GAN pipeline."""

    config: GANConfig
    trainer: GANTrainer
    evaluator: SupervisedEvaluator
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
    metrics: Dict[str, Metric]

    def __init__(
        self,
        config: GANConfig,
        checkpoint: Optional[str | PathLike] = None,
        checkpoint_keys: Optional[Sequence[str]] = None,
        logging: str = "online",
        wandb_id: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        group: Optional[str] = None,
        trainer_input_transform: Callable[[Any, Any], Any] = lambda x, y: (
            x,
            y,
        ),
        trainer_model_transform: Callable[[Any], Any] = lambda output: output,
        trainer_loss_transforms: Optional[
            Mapping[str, Callable[[Any, Any], Any]]
        ] = None,
        trainer_disc_loss_transforms: Optional[
            Mapping[str, Callable[[Any, Any], Any]]
        ] = None,
        trainer_disc_model_transform: Callable[
            [Any], Any
        ] = lambda output: output,
        trainer_output_transform: Callable[
            [Any, Any, Any, Any, Any], Any
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
        logging : str
            Logging mode. Must be either "online" or "offline".
        wandb_id : str
            W&B run ID to resume from.
        tags : list of str
            List of tags to add to the run in W&B.
        group : str
            Group to add run to in W&B.
        """

        # Parse config
        logging = logging.lower()
        assert logging in ("online", "offline")
        self.config = config

        # Create accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="wandb",
        )

        self._setup_tracking(
            mode=logging,
            wandb_id=wandb_id,
            tags=tags,
            group=group,
        )

        # Create datasets and data loaders
        self.datasets, self.loaders = self._create_data_loaders(
            batch_size=config.batch_size,
            loader_workers=config.loader_workers,
            datasets=config.datasets,
            loaders=config.loaders,
        )

        # Create model
        self.model = create_object_from_config(config.model)
        self.optimizer = create_object_from_config(
            config=config.optimizer,
            params=self.model.parameters(),
        )

        # Create discriminator model
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

        self.loss_fn = self._create_composite_loss(
            config.losses, trainer_loss_transforms
        )
        self.disc_loss_fn = self._create_composite_loss(
            config.disc_losses, trainer_disc_loss_transforms
        )
        self.trainer = GANTrainer(
            accelerator=self.accelerator,
            model=self.model,
            disc_model=self.disc_model,
            loss_fn=self.loss_fn,
            disc_loss_fn=self.disc_loss_fn,
            optimizer=self.optimizer,
            disc_optimizer=self.disc_optimizer,
            scheduler=self.lr_scheduler,
            disc_scheduler=self.disc_lr_scheduler,
            clip_grad_norm=config.clip_grad_norm,
            clip_grad_value=config.clip_grad_value,
            input_transform=trainer_input_transform,
            model_transform=trainer_model_transform,
            disc_model_transform=trainer_disc_model_transform,
            output_transform=trainer_output_transform,
            progress_label="train",
        )

        OutputLogger("train/loss", self.log, lambda o: o[0]).attach(
            self.trainer
        )
        OutputLogger("train/disc_loss", self.log, lambda o: o[1]).attach(
            self.trainer
        )
        CompositeLossLogger(self.loss_fn, self.log, "loss/").attach(
            self.trainer
        )
        CompositeLossLogger(self.disc_loss_fn, self.log, "disc_loss/").attach(
            self.trainer
        )
        OptimizerLogger(self.optimizer, ["lr"], self.log, "optimizer/").attach(
            self.trainer
        )
        OptimizerLogger(
            self.disc_optimizer, ["lr"], self.log, "disc_optimizer/"
        ).attach(self.trainer)

        # Create evaluator
        self.evaluator = SupervisedEvaluator(
            accelerator=self.accelerator,
            model=self.model,
            input_transform=evaluator_input_transform,
            model_transform=evaluator_model_transform,
            output_transform=evaluator_output_transform,
            progress_label="val",
        )

        if "val" in self.loaders:
            self.trainer.add_event_handler(
                event=self.log_interval,
                function=lambda: self.evaluator.run(self.loaders["val"]),
            )
        else:
            warnings.warn(
                'No "val" dataset provided.'
                " Validation will not be performed."
            )

        # Set up metric logging
        self.metrics = {}
        for metric_label, metric_conf in config.metrics.items():
            self.metrics[metric_label] = create_object_from_config(
                config=metric_conf,
                sync_on_compute=False,
            ).to(self.device)

        MetricLogger(
            metrics=self.metrics,
            log_function=self.log,
            prefix="val/",
        ).attach(self.evaluator)

        # Set up checkpoint handlers
        to_save = {
            "trainer": self.trainer,
            "model": self.model,
            "disc_model": self.disc_model,
            "optimizer": self.optimizer,
            "disc_optimizer": self.disc_optimizer,
            "scheduler": self.lr_scheduler,
            "disc_scheduler": self.disc_lr_scheduler,
        }
        to_unwrap = ["model", "disc_model"]
        output_folder = f"checkpoints/{self.run_name}"

        for ckpt_def in config.checkpoints:
            score_function = None
            if ckpt_def.metric is not None:
                score_function = partial(
                    lambda metric: metric.compute().item(),
                    metric=self.metrics[ckpt_def.metric],
                )

            checkpoint_handler = Checkpoint(
                accelerator=self.accelerator,
                config=self.config,
                to_save=to_save,
                output_folder=output_folder,
                global_step_function=lambda: self.trainer.iteration,
                score_function=score_function,
                score_name=ckpt_def.metric,
                score_mode=ckpt_def.mode,
                to_unwrap=to_unwrap,
                max_saved=ckpt_def.num_saved,
            )
            self.trainer.add_event_handler(
                event=parse_log_interval(ckpt_def.interval),
                function=checkpoint_handler,
            )

        # Load checkpoint
        if checkpoint is not None:
            self._load_checkpoint(
                path=checkpoint,
                to_load=to_save,
                to_unwrap=to_unwrap,
                keys=checkpoint_keys,
            )

    def run(self) -> None:
        """Run pipeline."""
        try:
            self.trainer.run(
                loader=self.loaders["train"],
                max_epochs=self.config.max_epochs,
            )
        except KeyboardInterrupt:
            self.print("Interrupted")
        finally:
            self.accelerator.end_training()
