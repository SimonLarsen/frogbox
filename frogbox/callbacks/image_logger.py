from collections.abc import Callable, Sequence
from typing import Any

import torch
import tqdm
from torch import Tensor
from torchvision.transforms.v2.functional import (
    InterpolationMode,
    center_crop,
    resize,
    to_pil_image,
)
from torchvision.utils import make_grid

from ..pipelines.pipeline import Pipeline
from ..tensor_utils import convert_tensor


def _default_forward(x: Any, y: Any, model: Callable) -> tuple[Any, ...]:
    return x, model(x), y


def _combine_images(
    images: Sequence[Tensor],
    resize_to_fit: bool = True,
    interpolation: str | InterpolationMode = "nearest-exact",
    num_cols: int | None = None,
) -> Tensor:
    for image in images:
        assert len(image.shape) == 3
        assert image.size(0) in (1, 3)

    max_h = max(image.size(1) for image in images)
    max_w = max(image.size(2) for image in images)

    transformed = []
    for image in images:
        c, h, w = image.shape
        if (h, w) != (max_h, max_w):
            if resize_to_fit:
                image = resize(
                    image,
                    size=[max_h, max_w],
                    interpolation=interpolation,
                )
            else:
                image = center_crop(image, output_size=[max_h, max_w])
        if c == 1:
            image = image.repeat((3, 1, 1))
        image = image.clamp(0.0, 1.0)
        transformed.append(image)

    return make_grid(
        tensor=transformed,
        normalize=False,
        nrow=num_cols or len(transformed),
    )


def log_images(
    pipeline: Pipeline,
    split: str = "test",
    model_key: str = "model",
    log_label: str = "images",
    resize_to_fit: bool = True,
    interpolation: str | InterpolationMode = "nearest-exact",
    show_progress: bool = False,
    num_cols: int | None = None,
    forward: Callable[[Any, Any, Callable], tuple[Any, ...]] | None = None,
) -> None:
    """
    Image logger callback.

    Parameters
    ----------
    model_key : str
        Pipeline model to use for inference.
    split : str
        Dataset split to evaluate on. Defaults to "test".
    log_label : str
        Label to log images under in Weights & Biases.
    resize_to_fit : bool
        If `true` smaller images are resized to fit canvas.
    interpolation : torchvision.transforms.functional.InterpolationMode
        Interpolation to use for resizing images.
    show_progress : bool
        Show progress bar.
    num_cols : int
        Number of columns in image grid.
        Defaults to number of elements in returned tuple.
    forward : callable
        Function that takes `x`, `y` and `model` and returns a tuple of images to log.
        Returns `(x, model(x), y)` if not provided.
    """
    if forward is None:
        forward = _default_forward

    model = pipeline._models[model_key]
    loader = pipeline._loaders[split]
    accelerator = pipeline.accelerator

    model.eval()

    data_iter = loader
    if show_progress:
        data_iter = tqdm.tqdm(
            data_iter,
            desc="Image logger",
            ncols=80,
            leave=False,
            total=len(data_iter),
        )

    images = []
    for batch in data_iter:
        x, y = batch

        with torch.inference_mode():
            outputs = forward(x, y, model)

        outputs = accelerator.gather_for_metrics(outputs)

        outputs = tuple(convert_tensor(e, torch.device("cpu")) for e in outputs)
        batch_sizes = [len(e) for e in outputs]
        assert all(s == batch_sizes[0] for s in batch_sizes)
        for i in range(batch_sizes[0]):
            grid = _combine_images(
                images=[e[i] for e in outputs],
                resize_to_fit=resize_to_fit,
                interpolation=interpolation,
                num_cols=num_cols,
            )
            images.append(grid)

    pil_images = [to_pil_image(image) for image in images]
    pipeline.log_images(log_label, pil_images)
