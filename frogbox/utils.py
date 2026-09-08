from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import cast

import torch

from .config import (
    Config,
    SupervisedConfig,
    create_object_from_config,
    read_config,
)


def load_model_checkpoint(
    path: str | PathLike,
    config_path: str | PathLike | None = None,
) -> tuple[torch.nn.Module, Config]:
    """
    Load model from checkpoint.

    Parameters
    ----------
    path : path
        Path to checkpoint file.
    config_path : path
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
    base_config = read_config(config_path)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    if base_config.type == "supervised":
        config = cast(SupervisedConfig, base_config)
        model = create_object_from_config(config.model)
        model.load_state_dict(ckpt["model"])
        return model, config
    else:
        raise RuntimeError(f"Unsupported config type {base_config.type}.")


def fix_compiled_model_keys(
    state_dict: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    fixed = {}
    for key, value in state_dict.items():
        key = key.replace("_orig_mod.", "")
        fixed[key] = value
    return fixed


def map_state_dict_to_compiled_model(
    state_dict: Mapping[str, torch.Tensor],
    model: torch.nn.Module,
) -> Mapping[str, torch.Tensor]:
    key_map = {key.replace("_orig_mod.", ""): key for key in model.state_dict()}
    fixed = {key_map[key]: value for key, value in state_dict.items()}
    return fixed
