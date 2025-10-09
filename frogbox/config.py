from typing import Any, Optional, Sequence, Mapping, Union, TypeAlias
from os import PathLike
from enum import Enum
from pathlib import Path
from functools import partial
import json
import yaml
from importlib import import_module
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
import jinja2
from .engines.events import EventStep, Event, MatchableEvent


_CONFIG_TYPE_EXTENSIONS: Mapping[str, Sequence[str]] = {
    "json": (".js", ".json"),
    "yaml": (".yml", ".yaml"),
}

_JINJA_EXTENSIONS: Sequence[str] = (".jinja", ".jinja2", ".j2")


class StrictModel(BaseModel):
    """@private"""
    model_config = ConfigDict(extra="forbid")


class ConfigType(str, Enum):
    """Pipeline configuration type."""

    SUPERVISED = "supervised"


class EngineType(str, Enum):
    TRAINER = "trainer"
    EVALUATOR = "evaluator"


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
    interval : EventStep or LogInterval
        Interval between saving checkpoints.
    num_saved : int
        Number of checkpoints to save.
    metric : str
        Name of metric to compare (optional).
    mode : CheckpointMode
        Whether to priority maximum or minimum metric value.
    """

    interval: EventStep | LogInterval = EventStep.EPOCH_COMPLETED
    num_saved: int = Field(default=3, ge=1)
    metric: Optional[str] = None
    mode: CheckpointMode = CheckpointMode.MAX


ObjectArgument: TypeAlias = Union["ObjectDefinition", Any]


class ObjectDefinition(StrictModel):
    object: Optional[str] = None
    function: Optional[str] = None
    args: Optional[Sequence[ObjectArgument]] = None
    kwargs: Optional[Mapping[str, ObjectArgument]] = None

    @model_validator(mode="after")
    def verify_object_or_function(self) -> "ObjectDefinition":
        if self.object is not None and self.function is not None:
            raise ValueError(
                "Object definition should only have either"
                ' "object" or "function" field.'
            )
        return self


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


class OptimizerDefinition(ObjectDefinition):
    """
    Optimizer definition.
    """

    class_name: str = "torch.optim.AdamW"
    kwargs: Optional[Mapping[str, ObjectArgument]] = {"lr": 1e-3}
    target: Optional[str] = None
    scheduler: LRSchedulerDefinition = LRSchedulerDefinition()


class ModelDefinition(ObjectDefinition):
    """
    Model definition.
    """

    optimizers: Mapping[str, OptimizerDefinition] = {
        "default": OptimizerDefinition()
    }


class LossDefinition(ObjectDefinition):
    """
    Loss function definition

    Attributes
    ----------
    weight : float
        Loss function weight.
    """

    weight: float = 1.0


class CallbackDefinition(ObjectDefinition):
    """
    Callback definition.
    """

    interval: EventStep | LogInterval = EventStep.EPOCH_COMPLETED
    engine: EngineType = EngineType.TRAINER


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
    batch_size : int
        Batch size.
    loader_workers : int
        How many subprocesses to use for data loading.
        `0` means the data will be loaded in the main process.
    max_epochs : int
        Maximum number of epochs to train for.
    gradient_accumulation_steps : int
        Number of steps the gradients should be accumulated across.
    checkpoints : list of CheckpointDefinition
    datasets : dict of ClassDefinition
        Dataset definitions.
    loaders : dict of ClassDefinition
        Data loader definitions.
    metrics : dict of ClassDefinition
        Evaluation metrics.
    """

    type: ConfigType
    project: str
    log_interval: EventStep | LogInterval = EventStep.EPOCH_COMPLETED
    batch_size: int = Field(default=32, ge=1)
    loader_workers: int = Field(default=0, ge=0)
    max_epochs: int = Field(default=50, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    checkpoints: Sequence[CheckpointDefinition] = (
        (
            CheckpointDefinition(
                metric=None,
                num_saved=3,
                interval=EventStep.EPOCH_COMPLETED,
            )
        ),
    )
    datasets: Mapping[str, ObjectDefinition]
    loaders: Mapping[str, ObjectDefinition] = {}
    metrics: Mapping[str, ObjectDefinition] = {}
    callbacks: Sequence[CallbackDefinition] = []

    @field_validator("datasets")
    @classmethod
    def check_datasets(cls, v: Mapping[str, ObjectDefinition]):
        assert "train" in v, 'datasets must contain key "train".'
        return v


class SupervisedConfig(Config):
    """
    Supervised pipeline configuration.

    Attributes
    ----------
    clip_grad_norm : float
        Clip gradients to norm if provided.
    clip_grad_norm : float
        Clip gradients to value if provided.
    model : ModelDefinition
        Model definition.
    losses : dict of LossDefinition
        Loss functions.
    """

    type: ConfigType = Field(default=ConfigType.SUPERVISED, frozen=True)
    clip_grad_norm: Optional[float] = None
    clip_grad_value: Optional[float] = None
    model: ModelDefinition
    losses: Mapping[str, LossDefinition] = {}
    trainer_forward: Optional[ObjectDefinition] = None
    evaluator_forward: Optional[ObjectDefinition] = None

    @field_validator("type")
    @classmethod
    def check_type(cls, v: ConfigType) -> ConfigType:
        assert v == ConfigType.SUPERVISED
        return v


def _guess_config_format(path: str | PathLike) -> str:
    """
    Guess file format based on filename.
    """
    path = str(path).lower()
    for filetype, exts in _CONFIG_TYPE_EXTENSIONS.items():
        for ext in exts:
            if path.endswith(ext):
                return filetype
            for jinja_ext in _JINJA_EXTENSIONS:
                if path.endswith(ext + jinja_ext):
                    return filetype
    raise ValueError(f"Cannot guess format of file {path}.")


def read_config(
    path: str | PathLike,
    format: Optional[str] = None,
    config_vars: Optional[Mapping[str, str]] = None,
) -> Config:
    """
    Read and render config file using jinja2.

    Parameters
    ----------
    path : str or path-like
        Path to JSON config file.
    format : str
        File format to read.
        If not provided, format will be inferred from filename.
    config_kwargs : str-to-str mapping
        Keyword arguments to pass to jinja2.
    """
    if config_vars is None:
        config_vars = {}

    if format is not None:
        format = format.lower()
        assert format in ("json", "yaml")
    else:
        format = _guess_config_format(path)

    # Read template
    path = Path(path)
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(path.parent)))
    template = env.get_template(str(path.relative_to(path.parent)))

    # Parse template
    data = template.render(config_vars)
    if format == "json":
        config = json.loads(data)
    else:
        config = yaml.safe_load(data)

    # Validate config object
    assert "type" in config
    if config["type"] == "supervised":
        return SupervisedConfig.model_validate(config)
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


def _get_module(path: str) -> Any:
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


def create_object_from_config(
    config: ObjectDefinition,
    *additional_args: Any,
    **additional_kwargs: Any,
) -> Any:
    """
    Create object from dictionary configuration.
    Dictionary should have a ``class`` entry and an optional ``params`` entry.

    Example
    -------
    >>> config = {"class": "torch.optim.Adam", "params": {"lr": 1e-5}}
    >>> optimizer = create_object_from_config(config)
    """
    # Recursively parse arguments
    args = []
    if config.args:
        for arg in config.args:
            if isinstance(arg, ObjectDefinition):
                arg = create_object_from_config(arg)
            args.append(arg)
    args.extend(additional_args)

    # Recursively parse keyword arguments
    kwargs = dict(additional_kwargs)
    if config.kwargs:
        for key, value in config.kwargs.items():
            if isinstance(value, ObjectDefinition):
                value = create_object_from_config(value)
            kwargs[key] = value

    if config.object is not None:
        obj_class = _get_module(config.object)
        return obj_class(*args, **kwargs)

    if config.function is not None:
        fun = _get_module(config.function)
        return partial(fun, *args, **kwargs)

    return RuntimeError("Cannot create object from definition.")
