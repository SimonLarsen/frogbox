from typing import cast, Union, Tuple, Any, Optional
from os import PathLike
from pathlib import Path
from .config import (
    read_json_config,
    Config,
    SupervisedConfig,
    create_object_from_config,
)
import torch


def load_model_checkpoint(
    path: Union[str, PathLike],
    config_path: Optional[Union[str, PathLike]] = None,
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
    else:
        raise RuntimeError(f"Unsupported config type {base_config.type}.")
