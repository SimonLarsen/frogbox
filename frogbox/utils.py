from typing import (
    cast,
    Tuple,
    Any,
    Optional,
    Sequence,
    Mapping,
    Callable,
)
from os import PathLike
from pathlib import Path
from .config import (
    read_json_config,
    Config,
    SupervisedConfig,
    GANConfig,
    create_object_from_config,
)
import torch


def load_model_checkpoint(
    path: str | PathLike,
    config_path: Optional[str | PathLike] = None,
) -> Tuple[Any, Config]:
    """
    Load model from checkpoint.

    Parameters
    ----------
    path : path-like
        Path to checkpoint file.
    config_path : path-like
        Path to config file. If empty config will be read from "config.json"
        in the same folder as `path`.

    Returns
    -------
    checkpoint : torch.nn.Module, Config
        Model checkpoint and config.
    """
    path = Path(path)
    if config_path is None:
        config_path = path.parent / "config.json"
    base_config = read_json_config(config_path)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    if base_config.type == "supervised":
        config = cast(SupervisedConfig, base_config)
        model = create_object_from_config(config.model)
        model.load_state_dict(ckpt["model"])
        return model, config
    elif base_config.type == "gan":
        config = cast(GANConfig, base_config)
        model = create_object_from_config(config.model)
        model.load_state_dict(ckpt["model"])
        return model, config
    else:
        raise RuntimeError(f"Unsupported config type {base_config.type}.")


def apply_to_tensor(x: Any, function: Callable) -> Any:
    """
    Recursively apply `function` to all tensors in collection.

    Parameters
    ----------
    x : tensor or collection containing tensors
        Object to apply function to.
    function : callable
        Function that takes a single tensors as argument.

    Returns
    -------
    A new collection of same type as `x`.
    """

    if isinstance(x, torch.Tensor):
        return function(x)
    if isinstance(x, (str, bytes)):
        return x
    if isinstance(x, Mapping):
        return cast(Callable, type(x))(
            {k: apply_to_tensor(sample, function) for k, sample in x.items()}
        )
    if isinstance(x, tuple) and hasattr(x, "_fields"):  # namedtuple
        return cast(Callable, type(x))(
            *(apply_to_tensor(sample, function) for sample in x)
        )
    if isinstance(x, Sequence):
        return cast(Callable, type(x))(
            [apply_to_tensor(sample, function) for sample in x]
        )
    raise TypeError(
        (f"x must contain tensors, dicts or lists; found {type(x)}")
    )


def convert_tensor(
    x: Any,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Any:
    """
    Recursively convert tensors in collection to `dtype` and/or move to
    `device`.

    Parameters
    ----------
    device : torch.device
        CUDA device to move tensors to.
    dtype : torch.dtype
        dtype to convert tensors to.
    """

    def _convert(e):
        return e.to(device=device, dtype=dtype)

    return apply_to_tensor(x, _convert)
