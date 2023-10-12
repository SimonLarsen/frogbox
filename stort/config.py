from typing import Union, Dict, Any, Optional
from os import PathLike
import warnings
from enum import Enum
from pathlib import Path
from importlib import import_module
import jinja2
import torch
from pydantic import BaseModel, field_validator
from ignite.engine import Events, CallableEventWithFilter
from ignite.handlers import (
    CosineAnnealingScheduler,
    create_lr_scheduler_with_warmup,
)


class LogEvent(str, Enum):
    EPOCH_COMPLETED = "EPOCH_COMPLETED"
    ITERATION_COMPLETED = "ITERATION_COMPLETED"
    COMPLETED = "COMPLETED"


class CheckpointMode(str, Enum):
    MIN = "min"
    MAX = "max"


class LogInterval(BaseModel):
    event: LogEvent
    every: int = 1


class ObjectDefinition(BaseModel):
    class_name: str
    params: Optional[Dict[str, Any]] = dict()


class LossDefinition(ObjectDefinition):
    weight: float


class SchedulerType(str, Enum):
    COSINE = "cosine"


class LRSchedulerDefinition(BaseModel):
    type: SchedulerType
    start_value: float = 1e-4
    end_value: float = 1e-7
    warmup_steps: int = 0


class Config(BaseModel):
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

    Example
    -------
        log_interval = {"event": "ITERATION_COMPLETED", "every": 100}
        event = parse_log_interval(log_interval)
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
        cl = get_class("torch.optim.Adam)
        optimizer = cl(lr=1e-5)
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
        config = {"class": "torch.optim.Adam", "params": {"lr": 1e-5}}
        optimizer = create_object_from_config(config)
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
    if config.type.lower() != "cosine":
        raise ValueError(f"Unsupported annealing schedule '{config.type}'.")

    lr_scheduler = CosineAnnealingScheduler(
        optimizer=optimizer,
        param_name="lr",
        start_value=config.start_value,
        end_value=config.end_value,
        cycle_size=max_iterations,
    )

    if config.warmup_steps > 0:
        lr_scheduler = create_lr_scheduler_with_warmup(
            lr_scheduler=lr_scheduler,
            warmup_start_value=0.0,
            warmup_end_value=config.start_value,
            warmup_duration=config.warmup_steps,
        )

    return lr_scheduler
