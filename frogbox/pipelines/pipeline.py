from typing import (
    Dict,
    Any,
    Optional,
    Mapping,
    Sequence,
    Callable,
    Tuple,
)
from os import PathLike
from abc import ABC
import datetime
import warnings
from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LRScheduler
from accelerate import Accelerator
from torchmetrics import Metric
from .name_generation import generate_name
from ..engines.engine import (
    Engine,
    Trainer,
    Evaluator,
    TrainerFactory,
    EvaluatorFactory,
)
from ..engines.events import MatchableEvent
from ..config import (
    Config,
    ClassDefinition,
    ModelDefinition,
    LossDefinition,
    parse_log_interval,
    create_object_from_config,
)
from .composite_loss import CompositeLoss
from .lr_scheduler import create_lr_scheduler
from ..handlers.checkpoint import Checkpoint
from ..handlers.composite_loss_logger import CompositeLossLogger
from ..handlers.optimizer_logger import OptimizerLogger
from ..handlers.metric_logger import MetricLogger


class Pipeline(ABC):
    """Pipeline abstract base class."""

    config: Config
    accelerator: Accelerator
    trainer: Trainer
    evaluator: Evaluator

    _models: Dict[str, torch.nn.Module]
    _optimizers: Dict[str, Dict[str, torch.optim.Optimizer]]
    _schedulers: Dict[str, Dict[str, LRScheduler]]
    _losses: Dict[str, CompositeLoss]
    _datasets: Dict[str, Dataset]
    _loaders: Dict[str, DataLoader]
    _metrics: Dict[str, Metric]

    _run_name: Optional[str] = None

    def __init__(
        self,
        config: Config,
        models: Mapping[str, ModelDefinition],
        losses: Mapping[str, Mapping[str, LossDefinition]],
        trainer: TrainerFactory,
        evaluator: EvaluatorFactory,
        checkpoint: Optional[str | PathLike] = None,
        checkpoint_keys: Optional[Sequence[str]] = None,
    ):
        self.config = config

        # Create accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="wandb",
        )
        self.accelerator.init_trackers(
            project_name=self.config.project,
            config=self.config.model_dump(),
            init_kwargs={"wandb": {"name": self.run_name}},
        )

        self._create_data_loaders(
            batch_size=config.batch_size,
            loader_workers=config.loader_workers,
            datasets=config.datasets,
            loaders=config.loaders,
        )
        self._create_models(models)
        self._create_losses(losses)
        self._create_metrics(config.metrics)

        self.trainer = trainer(
            self._models, self._optimizers, self._schedulers, self._losses
        )
        self.evaluator = evaluator(self._models)

        # Log trainer losses
        for name, loss in self._losses.items():
            CompositeLossLogger(
                loss=loss, log_function=self.log, prefix=f"loss/{name}/"
            ).attach(self.trainer)

        # Log learning rates
        for name1, optimizer_group in self._optimizers.items():
            for name2, optimizer in optimizer_group.items():
                OptimizerLogger(
                    optimizer=optimizer,
                    params=["lr"],
                    log_function=self.log,
                    prefix=f"optimizer/{name1}/{name2}/",
                ).attach(self.trainer)

        # Attach evaluator and log metrics
        if "val" in self._loaders:
            self.trainer.add_event_handler(
                event=self.log_interval,
                function=lambda: self.evaluator.run(
                    accelerator=self.accelerator,
                    loader=self._loaders["val"],
                    progress_label="val",
                ),
            )

            MetricLogger(
                metrics=self._metrics,
                log_function=self.log,
                prefix="metrics/",
            ).attach(self.evaluator)
        else:
            warnings.warn(
                'No "val" dataset provided. Validation will not be performed.'
            )

        # Create and attach checkpoints
        to_save, to_unwrap = self._get_checkpoint_dict()
        output_folder = f"checkpoints/{self.run_name}"
        for ckpt_cfg in config.checkpoints:
            score_function = None
            if ckpt_cfg.metric is not None:
                score_function = partial(
                    lambda metric: metric.compute().item(),
                    metric=self._metrics[ckpt_cfg.metric],
                )

            checkpoint_handler = Checkpoint(
                accelerator=self.accelerator,
                config=self.config,
                to_save=to_save,
                output_folder=output_folder,
                global_step_function=lambda: self.trainer.iteration,
                score_function=score_function,
                score_name=ckpt_cfg.metric,
                score_mode=ckpt_cfg.mode,
                to_unwrap=to_unwrap,
                max_saved=ckpt_cfg.num_saved,
            )
            self.trainer.add_event_handler(
                event=parse_log_interval(ckpt_cfg.interval),
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

    def _create_data_loaders(
        self,
        batch_size: int,
        loader_workers: int,
        datasets: Mapping[str, ClassDefinition],
        loaders: Optional[Mapping[str, ClassDefinition]] = None,
    ):
        if loaders is None:
            loaders = {}

        self._datasets = {}
        self._loaders = {}

        with self.accelerator.local_main_process_first():
            for split in datasets.keys():
                ds = create_object_from_config(datasets[split])
                self._datasets[split] = ds

                if split in loaders:
                    loader = create_object_from_config(
                        loaders[split],
                        dataset=ds,
                        batch_size=batch_size,
                        num_workers=loader_workers,
                    )
                else:
                    loader = DataLoader(
                        dataset=ds,
                        batch_size=batch_size,
                        num_workers=loader_workers,
                        shuffle=split == "train",
                    )

                loader = self.accelerator.prepare(loader)
                self._loaders[split] = loader

    def _create_models(self, models: Mapping[str, ModelDefinition]):
        self._models = {}
        self._optimizers = {}
        self._schedulers = {}

        for model_name, model_cfg in models.items():
            model = create_object_from_config(model_cfg)
            model = self.accelerator.prepare(model)

            self._models[model_name] = model
            self._optimizers[model_name] = {}
            self._schedulers[model_name] = {}

            for optimizer_name, optimizer_cfg in model_cfg.optimizers.items():
                optimizer = create_object_from_config(
                    optimizer_cfg,
                    params=model.parameters(),
                )

                scheduler = create_lr_scheduler(
                    optimizer=optimizer,
                    config=optimizer_cfg.scheduler,
                    max_iterations=self.max_iterations,
                )

                optimizer, scheduler = self.accelerator.prepare(
                    optimizer, scheduler
                )
                self._optimizers[model_name][optimizer_name] = optimizer
                self._schedulers[model_name][optimizer_name] = scheduler

    def _create_losses(
        self, losses: Mapping[str, Mapping[str, LossDefinition]]
    ):
        self._losses = {}
        for name, cfg in losses.items():
            self._losses[name] = self._create_composite_loss(cfg)

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
        ).to(self.device)
        return loss_fn

    def _create_metrics(
        self,
        config: Mapping[str, ClassDefinition],
    ) -> None:
        self._metrics = {}
        for label, conf in config.items():
            self._metrics[label] = create_object_from_config(
                config=conf,
                sync_on_compute=False,
            ).to(self.device)

    def _get_checkpoint_dict(self) -> Tuple[Mapping[str, Any], Sequence[str]]:
        to_save: Dict[str, Any] = {"trainer": self.trainer}
        to_unwrap = [f"model_{name}" for name in self._models]
        for name, model in self._models.items():
            to_save[f"model_{name}"] = model
        for name1, optimizer_group in self._optimizers.items():
            for name2, optimizer in optimizer_group.items():
                to_save[f"optimizer_{name1}_{name2}"] = optimizer
        for name1, scheduler_group in self._schedulers.items():
            for name2, scheduler in scheduler_group.items():
                to_save[f"scheduler_{name1}_{name2}"] = scheduler
        return to_save, to_unwrap

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

    def run(self) -> None:
        """Run pipeline."""
        try:
            self.trainer.run(
                accelerator=self.accelerator,
                loader=self._loaders["train"],
                max_epochs=self.config.max_epochs,
                progress_label="train",
            )
        except KeyboardInterrupt:
            self.print("Interrupted")
        finally:
            self.accelerator.end_training()

    def log(self, data: Mapping[str, Any]) -> None:
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

    @property
    def max_iterations(self) -> int:
        return len(self._loaders["train"]) * self.config.max_epochs
