from typing import Union, Dict, Any
from os import PathLike
from pathlib import Path
from importlib import import_module
import json
import jinja2
import torch
from ignite.engine import Events
from ignite.handlers import (
    CosineAnnealingScheduler,
    create_lr_scheduler_with_warmup,
)


def read_json_config(path: Union[str, PathLike]) -> Dict[str, Any]:
    """
    Read and render JSON config file and render using jinja2.
    """
    path = Path(path)
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(path.parent)))
    template = env.get_template(str(path.relative_to(path.parent)))
    config = json.loads(template.render())
    return config


def parse_log_interval(s: Union[str, Dict[str, Any]]) -> Events:
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
    event = Events[config.pop("event")]
    if len(config) > 0:
        event = event(**config)
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


def create_object_from_config(config: Dict[str, Any], **kwargs) -> Any:
    """
    Create object from dictionary configuration.
    Dictionary should have a ``class`` entry and an optional ``params`` entry.

    Example
    -------
        config = {"class": "torch.optim.Adam", "params": {"lr": 1e-5}}
        optimizer = create_object_from_config(config)
    """
    obj_class = get_class(config["class"])
    params = dict(config.get("params", {}))
    params.update(kwargs)
    return obj_class(**params)


def create_lr_scheduler_from_config(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    max_iterations: int,
) -> Any:
    """
    Create a learning rate scheduler from dictionary configuration.
    """
    if config["type"].lower() != "cosine":
        raise ValueError(f"Unsupported annealing schedule '{config['type']}'.")

    lr_scheduler = CosineAnnealingScheduler(
        optimizer=optimizer,
        param_name="lr",
        start_value=config["start_value"],
        end_value=config["end_value"],
        cycle_size=max_iterations,
    )

    warmup_steps = config.get("warmup_steps", 0)
    if warmup_steps > 0:
        lr_scheduler = create_lr_scheduler_with_warmup(
            lr_scheduler=lr_scheduler,
            warmup_start_value=0.0,
            warmup_end_value=config["start_value"],
            warmup_duration=warmup_steps,
        )

    return lr_scheduler
