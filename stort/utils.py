from typing import (
    Any,
    Iterable,
    Optional,
    Callable,
    Sequence,
    List,
    Union,
    Tuple,
    Dict,
)
from os import PathLike
from pathlib import Path
import torch
from torchvision.transforms.functional import (
    resize,
    center_crop,
    InterpolationMode,
)
from torchvision.utils import make_grid
from ignite.engine import _prepare_batch
from ignite.utils import convert_tensor
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
    A tuple (model, config).
    """
    path = Path(path)

    config_path = path.parent / "config.json"
    config = read_json_config(config_path)

    model = create_object_from_config(config.model)
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    return model, config


def predict_test_images(
    model: torch.nn.Module,
    data: Optional[Iterable],
    device: torch.device,
    prepare_batch: Callable = _prepare_batch,
    input_transform: Callable[[Any], Any] = lambda x: x,
    model_transform: Callable[[Any], Any] = lambda y_pred: y_pred,
    output_transform: Callable[[Any, Any, Any], Any] = (
        lambda x, y, y_pred: [x, y_pred, y]
    ),
    resize_to_fit: bool = True,
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    antialias: bool = True,
    num_cols: Optional[int] = None,
    amp_mode: Optional[str] = None,
    non_blocking: bool = False,
) -> List[torch.Tensor]:
    model.eval()

    images = []
    for batch in iter(data):
        x, y = prepare_batch(batch, device, non_blocking)
        x = input_transform(x)
        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type, enabled=amp_mode == "amp"
            ):
                y_pred = model_transform(model(x))

        y_pred = y_pred.type(y.dtype)
        x, y, y_pred = convert_tensor(
            x=(x, y, y_pred),
            device=torch.device("cpu"),
            non_blocking=non_blocking,
        )

        output = output_transform(x, y, y_pred)
        batch_sizes = [len(e) for e in output]
        assert all(s == batch_sizes[0] for s in batch_sizes)

        for i in range(batch_sizes[0]):
            grid = _combine_test_images(
                images=[e[i] for e in output],
                resize_to_fit=resize_to_fit,
                interpolation=interpolation,
                antialias=antialias,
                num_cols=num_cols,
            )
            images.append(grid)
    return images


def _combine_test_images(
    images: Sequence[torch.Tensor],
    resize_to_fit: bool = True,
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    antialias: bool = True,
    num_cols: Optional[int] = None,
) -> torch.Tensor:
    for image in images:
        assert len(image.shape) == 3
        assert image.size(0) in (1, 3)

    max_h = max(image.size(1) for image in images)
    max_w = max(image.size(2) for image in images)

    transformed = []
    for image in images:
        C, H, W = image.shape
        if H != max_h or W != max_w:
            if resize_to_fit:
                image = resize(
                    image,
                    size=(max_h, max_w),
                    interpolation=interpolation,
                    antialias=antialias,
                )
            else:
                image = center_crop(image, output_size=(max_h, max_w))
        if C == 1:
            image = image.repeat((3, 1, 1))
        image = image.clamp(0.0, 1.0)
        transformed.append(image)

    if len(transformed) == 1:
        return transformed[0]
    else:
        return make_grid(
            tensor=transformed,
            normalize=False,
            nrow=num_cols or len(transformed),
        )
