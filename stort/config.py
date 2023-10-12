from typing import Union, Dict, Any, Optional
from os import PathLike
import warnings
from enum import Enum
from pathlib import Path
from math import ceil
from importlib import import_module
import jinja2
import torch
from pydantic import BaseModel, field_validator
from ignite.engine import Events, CallableEventWithFilter
from ignite.handlers import (
    CosineAnnealingScheduler,
    LinearCyclicalScheduler,
    create_lr_scheduler_with_warmup,
)


class LogEvent(str, Enum):
    """
    Log event.
    """

    EPOCH_COMPLETED = "EPOCH_COMPLETED"
    ITERATION_COMPLETED = "ITERATION_COMPLETED"
    COMPLETED = "COMPLETED"


class CheckpointMode(str, Enum):
    """
    Checkpoint evaluation mode.
    """

    MIN = "min"
    MAX = "max"


class LogInterval(BaseModel):
    """
    Logging interval.

    Attributes
    ----------
    event : LogEvent
        Event trigger.
    interval : int
        How often event should trigger. Defaults to every time (`1`).
    """

    event: LogEvent
    every: int = 1


class ObjectDefinition(BaseModel):
    """
    Object instance definition.

    Attributes
    ----------
    class_name : str
        Class path string. Example: `torch.optim.AdamW`.
    params : dict
        Dictionary of parameters to pass object constructor.
    """

    class_name: str
    params: Optional[Dict[str, Any]] = dict()


class LossDefinition(ObjectDefinition):
    """
    Loss function definition

    Attributes
    ----------
    weight : float
        Loss function weight.
    """

    weight: float = 1.0


class SchedulerType(str, Enum):
    """
    Parameter scheduler type.
    """

    LINEAR = "linear"
    COSINE = "cosine"


class LRSchedulerDefinition(BaseModel):
    """
    Learning rate scheduler definition.

    Attributes
    ----------
    type : SchedulerType
        Scheduler type.
    start_value : float
        Initial learning rate.
    end_value : float
        Final learning rate.
    cycles : int
        Number of scheduler cycles. Defaults to `1`.
    start_value_mult : float
        Ratio by which to change the start value at the end of each cycle.
    end_value_mult : float
        Ratio by which to change the end value at the end of each cycle.
    warmup_stets : int
        Number of steps to perform warmup. Set to `0` to disable warmup.
    """

    type: SchedulerType = SchedulerType.COSINE
    start_value: float = 3e-4
    end_value: float = 1e-7
    cycles: int = 1
    start_value_mult: float = 1.0
    end_value_mult: float = 1.0
    warmup_steps: int = 0


class Config(BaseModel):
    """
    Trainer configuration.

    Attributes
    ----------
    project : str
        Project name.
    amp : bool
        If `true` automatic mixed-precision is enabled.
    clip_grad_norm : float
        Clip gradients to norm if provided.
    batch_size : int
        Batch size.
    loader_workers : int
        How many subprocesses to use for data loading.
        `0` means the data will be loaded in the main process.
    max_epochs : int
        Maximum number of epochs to train for.
    checkpoint_metric : str
        Name of metric to use for evaluating checkpoints.
    checkpoint_mode : CheckpointMode
        Either `min` or `max`. Determines whether to keep the checkpoints
        with the greatest or lowest metric score.
    checkpoint_n_saved : int
        Number of checkpoints to keep.
    log_interval : str or LogInterval
        At which interval to log metrics.
    model : ObjectDefinition
        Model object definition.
    losses : dict of LossDefinition
        Loss functions.
    metrics : dict of ObjectDefinition
        Evaluation metrics.
    datasets : dict of ObjectDefinition
        Datasets.
    optimizer : ObjectDefinition
        Torch optimizer.
    lr_scheduler : LRSchedulerDefinition
        Learning rate scheduler.
    meta : dict
        Additional meta data.
    """

    project: str
    amp: bool = False
    clip_grad_norm: Optional[float] = None
    batch_size: int = 32
    loader_workers: int = 0
    max_epochs: int = 32
    checkpoint_metric: str
    checkpoint_mode: CheckpointMode = CheckpointMode.MAX
    checkpoint_n_saved: int = 3
    log_interval: Union[str, LogInterval] = LogInterval(
        event=LogEvent.EPOCH_COMPLETED, every=1
    )
    model: ObjectDefinition
    losses: Dict[str, LossDefinition]
    metrics: Dict[str, ObjectDefinition]
    datasets: Dict[str, ObjectDefinition]
    optimizer: ObjectDefinition
    lr_scheduler: LRSchedulerDefinition
    meta: Dict[str, Any] = dict()

    @field_validator("datasets")
    @classmethod
    def validate_datasets(cls, v):
        assert "train" in v, "'train' missing in datasets definition."
        assert "val" in v, "'val' missing in datasets definition."
        return v

    @field_validator("losses")
    @classmethod
    def validate_losses(cls, v):
        if len(v) == 0:
            warnings.warn("No loss functions defined.")
        return v


def read_json_config(path: Union[str, PathLike]) -> Config:
    """
    Read and render JSON config file and render using jinja2.

    Parameters
    ----------
    path : str or path-like
        Path to JSON config file.
    """
    path = Path(path)
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(path.parent)))
    template = env.get_template(str(path.relative_to(path.parent)))
    config = Config.model_validate_json(template.render())
    return config


def parse_log_interval(
    s: Union[str, LogInterval]
) -> Union[Events, CallableEventWithFilter]:
    """
    Create ignite event from string or dictionary configuration.
    Dictionary must have a ``event`` entry.
    """
    if isinstance(s, str):
        return Events[s]
    config = dict(s)
    for k, v in config.items():
        if v is None:
            del config[k]

    event_name = config.pop("event")
    event = Events[event_name.value]

    if len(config) > 0:
        return event(**config)
    return event


def get_class(path: str) -> Any:
    """
    Get class from import path.

    Example
    -------
    >>> cl = get_class("torch.optim.Adam)
    >>> optimizer = cl(lr=1e-5)
    """
    parts = path.split(".")
    module_path = ".".join(parts[:-1])
    class_name = parts[-1]
    module = import_module(module_path)
    return getattr(module, class_name)


def create_object_from_config(config: ObjectDefinition, **kwargs) -> Any:
    """
    Create object from dictionary configuration.
    Dictionary should have a ``class`` entry and an optional ``params`` entry.

    Example
    -------
    >>> config = {"class": "torch.optim.Adam", "params": {"lr": 1e-5}}
    >>> optimizer = create_object_from_config(config)
    """
    obj_class = get_class(config.class_name)
    params = dict(config.params)
    params.update(kwargs)
    return obj_class(**params)


def create_lr_scheduler_from_config(
    optimizer: torch.optim.Optimizer,
    config: LRSchedulerDefinition,
    max_iterations: int,
) -> Any:
    """
    Create a learning rate scheduler from dictionary configuration.
    """
    cycle_size = ceil(max_iterations / config.cycles)

    if config.type == SchedulerType.COSINE:
        lr_scheduler = CosineAnnealingScheduler(
            optimizer=optimizer,
            param_name="lr",
            start_value=config.start_value,
            end_value=config.end_value,
            cycle_size=cycle_size,
            start_value_mult=config.start_value_mult,
            end_value_mult=config.end_value_mult,
        )
    elif config.type == SchedulerType.LINEAR:
        lr_scheduler = LinearCyclicalScheduler(
            optimizer=optimizer,
            param_name="lr",
            start_value=config.start_value,
            end_value=config.end_value,
            cycle_size=cycle_size,
            start_value_mult=config.start_value_mult,
            end_value_mult=config.end_value_mult,
        )
    else:
        raise RuntimeError(f'Unsupported LR scheduler "{config.type}".')

    if config.warmup_steps > 0:
        lr_scheduler = create_lr_scheduler_with_warmup(
            lr_scheduler=lr_scheduler,
            warmup_start_value=0.0,
            warmup_end_value=config.start_value,
            warmup_duration=config.warmup_steps,
        )

    return lr_scheduler
