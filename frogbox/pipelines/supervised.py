from typing import Dict, Optional, Sequence, Callable, Any, Mapping
from os import PathLike
import warnings
from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from torchmetrics import Metric
from .pipeline import Pipeline
from ..config import (
    SupervisedConfig,
    create_object_from_config,
    parse_log_interval,
)
from ..engines.supervised import SupervisedTrainer, SupervisedEvaluator
from .lr_scheduler import create_lr_scheduler
from .composite_loss import CompositeLoss
from ..handlers.output_logger import OutputLogger
from ..handlers.metric_logger import MetricLogger
from ..handlers.optimizer_logger import OptimizerLogger
from ..handlers.composite_loss_logger import CompositeLossLogger
from ..handlers.checkpoint import Checkpoint


class SupervisedPipeline(Pipeline):
    """Supervised pipeline."""

    config: SupervisedConfig
    trainer: SupervisedTrainer
    evaluator: SupervisedEvaluator
    datasets: Dict[str, Dataset]
    loaders: Dict[str, DataLoader]
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    loss_fn: CompositeLoss
    metrics: Dict[str, Metric]

    def __init__(
        self,
        config: SupervisedConfig,
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
        trainer_output_transform: Callable[
            [Any, Any, Any, Any], Any
        ] = lambda x, y, y_pred, loss: loss.item(),
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
        Create supervised pipeline.

        Parameters
        ----------
        config : SupervisedConfig
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
        trainer_input_transform : callable
            Function that receives tensors `x` and `y` and outputs tuple of
            tensors `(x, y)`.
        trainer_model_transform : callable
            Function that receives the output from the model during training
            and converts it into the form as required by the loss function.
        trainer_loss_transforms : mapping of callables
            Dictionary of functions that transform the inputs passed to each
            loss function. Each function receives `y_pred` and `y`.
            Default is returning `(y_pred, y)`.
        trainer_output_transform : callable
            Function that receives `x`, `y`, `y_pred`, `loss` and returns value
            to be assigned to trainer's `state.output` after each iteration.
            Default is returning `loss.item()`.
        evaluator_input_transform : callable
            Function that receives tensors `x` and `y` and outputs tuple of
            tensors `(x, y)`.
        evaluator_model_transform : callable
            Function that receives the output from the model during evaluation
            and converts it into the predictions:
            `y_pred = model_transform(model(x))`.
        evaluator_output_transform : callable
            Function that receives `x`, `y`, `y_pred` and returns value to be
            passed to output handlers after each iteration.
            Default is returning `(y_pred, y)` which fits output expected by
            metrics.
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

        # Create learning rate scheduler
        max_iterations = len(self.loaders["train"]) * config.max_epochs
        self.lr_scheduler = create_lr_scheduler(
            optimizer=self.optimizer,
            config=config.lr_scheduler,
            max_iterations=max_iterations,
        )

        # Wrap with accelerator
        self.model, self.optimizer, self.lr_scheduler = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.lr_scheduler
            )
        )
        for split in self.loaders.keys():
            self.loaders[split] = self.accelerator.prepare(self.loaders[split])

        # Create trainer
        self.loss_fn = self._create_composite_loss(
            config.losses, trainer_loss_transforms
        )
        self.trainer = SupervisedTrainer(
            accelerator=self.accelerator,
            model=self.model,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            clip_grad_norm=config.clip_grad_norm,
            clip_grad_value=config.clip_grad_value,
            input_transform=trainer_input_transform,
            model_transform=trainer_model_transform,
            output_transform=trainer_output_transform,
            progress_label="train",
        )

        OutputLogger("train/loss", self.log).attach(self.trainer)
        CompositeLossLogger(self.loss_fn, self.log, "loss/").attach(
            self.trainer
        )
        OptimizerLogger(self.optimizer, ["lr"], self.log, "optimizer/").attach(
            self.trainer
        )

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
            "optimizer": self.optimizer,
            "scheduler": self.lr_scheduler,
        }
        to_unwrap = ["model"]
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
