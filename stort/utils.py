"""
## Loading a trained model

Trained models can be loaded with `stort.utils.load_model_checkpoint`. The function returns the trained model as well the trainer configuration.

```python
import torch
from stort.utils import load_model_checkpoint

device = torch.device("cuda:0")

model, config = load_model_checkpoint(
    "checkpoints/smooth-jazz-123/best_checkpoint_1_PSNR=26.6363.pt"
)
model = model.eval().to(device)

x = torch.rand((1, 3, 16, 16), device=device)
with torch.inference_mode():
    pred = model(x)
```
"""
from typing import Union, Tuple
from os import PathLike
from pathlib import Path
import torch
from .config import (
    Config,
    read_json_config,
    create_object_from_config,
)


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

    if config.type == "supervised":
        model = create_object_from_config(config.model)
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        return model, config
    else:
        raise RuntimeError(f"Unsupported config type {config.type}.")
