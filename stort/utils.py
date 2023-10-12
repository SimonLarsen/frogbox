from typing import Union, Tuple
from os import PathLike
from pathlib import Path
import torch
from .config import Config, read_json_config, create_object_from_config


def load_model_checkpoint(
    path: Union[str, PathLike]
) -> Tuple[torch.nn.Module, Config]:
    """
    Load model from checkpoint.

    Parameters
    ----------
    path : path-like
        Path to checkpoint file.

    Returns
    -------
    checkpoint : torch.nn.Module, Config
        Model checkpoint and config.
    """
    path = Path(path)

    config_path = path.parent / "config.json"
    config = read_json_config(config_path)

    model = create_object_from_config(config.model)
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    return model, config
