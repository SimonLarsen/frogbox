from typing import Dict, Any, Optional, Sequence
from os import PathLike
from enum import Enum
from pathlib import Path
import json
from pydantic import BaseModel, ConfigDict, Field, field_validator
import jinja2
from importlib import import_module
from .engines.events import EventStep, Event, MatchableEvent


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConfigType(str, Enum):
    """Pipeline configuration type."""

    SUPERVISED = "supervised"
    GAN = "gan"


class LogInterval(StrictModel):
    """
    Logging interval.

    Attributes
    ----------
    event : Events
        Event trigger.
    interval : int
        How often event should trigger. Defaults to every time (`1`).
    first : int
        First step where event should trigger (zero-indexed).
    last : int
        Last step where vent should trigger (zero-indexed).
    """

    event: EventStep
    every: int = 1
    first: Optional[int] = None
    last: Optional[int] = None


class CheckpointMode(str, Enum):
    """Checkpoint evaluation mode."""

    MIN = "min"
    MAX = "max"


class CheckpointDefinition(StrictModel):
    """
    Checkpoint definition.


    Attributes
    ----------
    metric : str
        Name of metric to compare (optional).
    mode : CheckpointMode
        Whether to priority maximum or minimum metric value.
    num_saved : int
        Number of checkpoints to save.
    interval : EventStep or LogInterval
        Interval between saving checkpoints.
    """

    metric: Optional[str] = None
    mode: CheckpointMode = CheckpointMode.MAX
    num_saved: int = Field(default=3, ge=1)
    interval: EventStep | LogInterval = EventStep.EPOCH_COMPLETED


class ObjectDefinition(StrictModel):
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
    params: Dict[str, Any] = dict()


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


class LRSchedulerDefinition(StrictModel):
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
    end_value: float = 1e-7
    warmup_steps: int = Field(default=0, ge=0)


class Config(StrictModel):
    """
    Base configuration.

    Attributes
    ----------
    type : ConfigType
        Pipeline type.
    project : str
        Project name.
    log_interval : EventStep or LogInterval
        At which interval to log metrics.
    """

    type: ConfigType
    project: str
    meta: Dict[str, Any] = dict()
    log_interval: EventStep | LogInterval = EventStep.EPOCH_COMPLETED


class SupervisedConfig(Config):
    """
    Supervised pipeline configuration.

    Attributes
    ----------
    batch_size : int
        Batch size.
    loader_workers : int
        How many subprocesses to use for data loading.
        `0` means the data will be loaded in the main process.
    max_epochs : int
        Maximum number of epochs to train for.
    clip_grad_norm : float
        Clip gradients to norm if provided.
    clip_grad_norm : float
        Clip gradients to value if provided.
    gradient_accumulation_steps : int
        Number of steps the gradients should be accumulated across.
    datasets : dict of ObjectDefinition
        Dataset definitions.
    loaders : dict of ObjectDefinition
        Data loader definitions.
    model : ObjectDefinition
        Model definition.
    losses : dict of LossDefinition
        Loss functions.
    metrics : dict of ObjectDefinition
        Evaluation metrics.
    optimizer : ObjectDefinition
        Torch optimizer.
    lr_scheduler : LRSchedulerDefinition
        Learning rate scheduler.
    """

    batch_size: int = Field(default=32, ge=1)
    loader_workers: int = Field(default=0, ge=0)
    max_epochs: int = Field(default=50, ge=1)
    clip_grad_norm: Optional[float] = None
    clip_grad_value: Optional[float] = None
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    metrics: Dict[str, ObjectDefinition] = dict()
    checkpoints: Sequence[CheckpointDefinition] = (
        (
            CheckpointDefinition(
                metric=None,
                num_saved=3,
                interval=EventStep.EPOCH_COMPLETED,
            )
        ),
    )
    model: ObjectDefinition
    losses: Dict[str, LossDefinition] = dict()
    datasets: Dict[str, ObjectDefinition]
    loaders: Dict[str, ObjectDefinition] = dict()
    optimizer: ObjectDefinition = ObjectDefinition(
        class_name="torch.optim.AdamW",
        params={"lr": 1e-3},
    )
    lr_scheduler: LRSchedulerDefinition = LRSchedulerDefinition()

    @field_validator("type")
    @classmethod
    def check_type(cls, v: ConfigType) -> ConfigType:
        assert v == ConfigType.SUPERVISED
        return v

    @field_validator("datasets")
    @classmethod
    def check_datasets(cls, v: Dict[str, ObjectDefinition]):
        assert "train" in v, 'datasets must contain key "train".'
        return v


class GANConfig(SupervisedConfig):
    """
    GAN pipeline configuration.

    Attributes
    ----------
    disc_model : ObjectDefinition
        Discriminator model definition.
    disc_losses: dict of LossDefinition
        Discriminator loss functions.
    disc_optimizer : ObjectDefinition
        Discriminator Torch optimizer.
    disc_lr_scheduler : LRSchedulerDefinition
        Discriminator learning rate scheduler.
    """

    disc_model: ObjectDefinition
    disc_losses: Dict[str, LossDefinition] = dict()
    disc_optimizer: ObjectDefinition = ObjectDefinition(
        class_name="torch.optim.AdamW",
        params={"lr": 1e-3},
    )
    disc_lr_scheduler: LRSchedulerDefinition = LRSchedulerDefinition()

    @field_validator("type")
    @classmethod
    def check_type(cls, v: ConfigType) -> ConfigType:
        assert v == ConfigType.GAN
        return v


def read_json_config(path: str | PathLike) -> Config:
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
    config = json.loads(template.render())
    assert "type" in config
    if config["type"] == "supervised":
        return SupervisedConfig.model_validate(config)
    elif config["type"] == "gan":
        return GANConfig.model_validate(config)
    else:
        raise RuntimeError(f"Unknown config type {config['type']}.")


def parse_log_interval(e: str | LogInterval) -> MatchableEvent:
    """Create matchable event from log interval configuration."""
    if isinstance(e, str):
        return Event(event=e)

    if isinstance(e, LogInterval):
        return Event(
            event=e.event,
            every=e.every,
            first=e.first,
            last=e.last,
        )

    raise ValueError(f"Cannot parse log interval {e}.")


def _get_class(path: str) -> Any:
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
    obj_class = _get_class(config.class_name)
    params = dict(kwargs)
    params.update(config.params)
    return obj_class(**params)
