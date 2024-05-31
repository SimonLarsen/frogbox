"""
# Logging images

The simplest way to log images during training is to create an callback with
`frogbox.callbacks.image_logger.create_image_logger`:

```python
from frogbox import Events
from frogbox.callbacks import create_image_logger

pipeline.install_callback(
    event=Events.EPOCH_COMPLETED,
    callback=create_image_logger(),
)
```

Images can automatically be denormalized by setting `denormalize_input`/`denormalize_output`
and providing the mean and standard deviation used for normalization.

For instance, if input images are normalized with ImageNet parameters and outputs are in [0, 1]:

```python
image_logger = create_image_logger(
    normalize_mean=[0.485, 0.456, 0.406],
    normalize_std=[0.229, 0.224, 0.225],
    denormalize_input=True,
)
```

More advanced transformations can be made by overriding `input_transform`, `model_transform`, or `output_transform`:

```python
from torchvision.transforms.functional import hflip

def flip_input(x, y, y_pred):
    x = hflip(x)
    return x, y_pred, y

image_logger = create_image_logger(
    output_transform=flip_input,
)
```
"""  # noqa: E501

from typing import Callable, Any, Sequence, Optional
import torch
from torchvision.transforms.functional import (
    InterpolationMode,
    resize,
    center_crop,
    to_pil_image,
)
from torchvision.utils import make_grid
from ignite.engine import _prepare_batch
from ignite.utils import convert_tensor
from kornia.enhance import Denormalize
import wandb
import tqdm
from ..pipelines.supervised import SupervisedPipeline


def create_image_logger(
    split: str = "test",
    log_label: str = "test/images",
    resize_to_fit: bool = True,
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    antialias: bool = True,
    num_cols: Optional[int] = None,
    denormalize_input: bool = False,
    denormalize_target: bool = False,
    normalize_mean: Sequence[float] = (0.0, 0.0, 0.0),
    normalize_std: Sequence[float] = (1.0, 1.0, 1.0),
    progress: bool = False,
    prepare_batch: Callable = _prepare_batch,
    input_transform: Callable[[Any, Any], Any] = lambda x, y: (x, y),
    model_transform: Callable[[Any], Any] = lambda output: output,
    output_transform: Callable[[Any, Any, Any], Any] = lambda x, y, y_pred: (
        x,
        y_pred,
        y,
    ),
):
    """
    Create image logger callback.

    Parameters
    ----------
    split : str
        Dataset split to evaluate on. Defaults to "test".
    log_label : str
        Label to log images under in Weights & Biases.
    resize_to_fit : bool
        If `true` smaller images are resized to fit canvas.
    interpolation : torchvision.transforms.functional.InterpolationMode
        Interpolation to use for resizing images.
    antialias : bool
        If `true` antialiasing is used when resizing images.
    num_cols : int
        Number of columns in image grid.
        Defaults to number of elements in returned tuple.
    denormalize_input : bool
        If `true` input images (x) a denormalized after inference.
    denormalize_target : bool
        If `true` target images (y and y_pred) are denormalized after inference.
    normalize_mean : (float, float, float)
        RGB mean values used in image normalization.
    normalize_std : (float, float, float)
        RGB std.dev. values used in image normalization.
    progress : bool
        Show progress bar.
    prepare_batch : Callable
        Function that receives `batch`, `device`, `non_blocking` and
        outputs tuple of tensors `(batch_x, batch_y)`.
    input_transform : Callable
        Function that receives tensors `y` and `y` and outputs tuple of
        tensors `(x, y)`.
    model_transform : Callable
        Function that receives the output from the model during evaluation
        and converts it into the predictions:
        `y_pred = model_transform(model(x))`.
    output_transform : Callable
        Function that receives `x`, `y`, `y_pred` and returns tensors to be
        logged as images. Default is returning `(x, y_pred, y)`.
    """  # noqa: E501
    denormalize = Denormalize(
        torch.as_tensor(normalize_mean),
        torch.as_tensor(normalize_std),
    )

    def _callback(pipeline: SupervisedPipeline):
        model = pipeline.model
        config = pipeline.config
        device = pipeline.device
        loaders = pipeline.loaders
        trainer = pipeline.trainer

        model.eval()

        data_iter = iter(loaders[split])
        if progress:
            data_iter = tqdm.tqdm(
                data_iter,
                desc="Images",
                ncols=80,
                leave=False,
                total=len(loaders[split]),
            )

        images = []
        for batch in data_iter:
            x, y = prepare_batch(batch, device, non_blocking=False)
            x, y = input_transform(x, y)
            with torch.inference_mode():
                with torch.autocast(
                    device_type=device.type, enabled=config.amp
                ):
                    y_pred = model_transform(model(x))

                x, y, y_pred = convert_tensor(  # type: ignore
                    x=(x, y, y_pred),
                    device=torch.device("cpu"),
                    non_blocking=False,
                )

                if denormalize_input:
                    x = denormalize(x)
                if denormalize_target:
                    y = denormalize(y)
                    y_pred = denormalize(y_pred)

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

        wandb_images = [wandb.Image(to_pil_image(image)) for image in images]
        wandb.log(step=trainer.state.iteration, data={log_label: wandb_images})

    return _callback


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
