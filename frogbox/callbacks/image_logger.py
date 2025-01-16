"""
# Logging images

The simplest way to log images during training is to create a callback with
`frogbox.callbacks.image_logger.ImageLogger`:

```python
from frogbox import Events
from frogbox.callbacks import ImageLogger

image_logger = ImageLogger()

pipeline.install_callback(
    event="epoch_completed",
    callback=image_logger,
)
```

Images can automatically be denormalized by setting `denormalize_input`/`denormalize_output`
and providing the mean and standard deviation used for normalization.

For instance, if input images are normalized with ImageNet parameters and outputs are in [0, 1]:

```python
image_logger = ImageLogger(
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

image_logger = ImageLogger(
    output_transform=flip_input,
)
```
"""  # noqa: E501

from typing import Sequence, Callable, Any, Optional
import torch
from torchvision.transforms.functional import (
    center_crop,
    resize,
    InterpolationMode,
    to_pil_image,
)
from torchvision.utils import make_grid
import tqdm
import wandb
from .callback import Callback
from ..pipelines.pipeline import Pipeline
from ..utils import convert_tensor


class ImageLogger(Callback):
    """Callback for logging images."""

    def __init__(
        self,
        split: str = "test",
        log_label: str = "test/images",
        resize_to_fit: bool = True,
        interpolation: str | InterpolationMode = "nearest",
        num_cols: Optional[int] = None,
        denormalize_input: bool = False,
        denormalize_target: bool = False,
        normalize_mean: Sequence[float] = (0.0, 0.0, 0.0),
        normalize_std: Sequence[float] = (1.0, 1.0, 1.0),
        show_progress: bool = False,
        input_transform: Callable[[Any, Any], Any] = lambda x, y: (x, y),
        model_transform: Callable[[Any], Any] = lambda output: output,
        output_transform: Callable[
            [Any, Any, Any], Any
        ] = lambda x, y, y_pred: (x, y_pred, y),
    ):
        """
        Create ImageLogger.

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
        num_cols : int
            Number of columns in image grid.
            Defaults to number of elements in returned tuple.
        denormalize_input : bool
            If `true` input images (x) a denormalized after inference.
        denormalize_target : bool
            If `true` target images (y and y_pred) are denormalized after
            inference.
        normalize_mean : (float, float, float)
            RGB mean values used in image normalization.
        normalize_std : (float, float, float)
            RGB std.dev. values used in image normalization.
        show_progress : bool
            Show progress bar.
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
        """
        self._split = split
        self._log_label = log_label
        self._resize_to_fit = resize_to_fit
        self._interpolation = InterpolationMode(interpolation)
        self._num_cols = num_cols
        self._denormalize_input = denormalize_input
        self._denormalize_target = denormalize_target
        self._normalize_mean = normalize_mean
        self._normalize_std = normalize_std
        self._show_progress = show_progress
        self._input_transform = input_transform
        self._model_transform = model_transform
        self._output_transform = output_transform

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(
            self._normalize_mean, device=x.device, dtype=x.dtype
        ).reshape(1, -1, 1, 1)

        std = torch.as_tensor(
            self._normalize_std, device=x.device, dtype=x.dtype
        ).reshape(1, -1, 1, 1)

        return (x * std) + mean

    def __call__(self, pipeline: Pipeline) -> None:
        if not hasattr(pipeline, "model") or not hasattr(pipeline, "loaders"):
            raise RuntimeError(
                f"ImageLogger not compatible with pipeline {pipeline}."
            )

        model = pipeline.model
        loaders = pipeline.loaders
        accelerator = pipeline.accelerator

        model.eval()

        data_iter = loaders[self._split]
        if self._show_progress:
            data_iter = tqdm.tqdm(
                data_iter,
                desc="Images",
                ncols=80,
                leave=False,
                total=len(data_iter),
            )

        images = []
        for batch in data_iter:
            x, y = batch
            x, y = self._input_transform(x, y)

            with torch.inference_mode():
                y_pred = self._model_transform(model(x))

            x, y, y_pred = accelerator.gather_for_metrics((x, y, y_pred))

            x = convert_tensor(x, device=torch.device("cpu"))
            y = convert_tensor(y, device=torch.device("cpu"))
            y_pred = convert_tensor(y_pred, device=torch.device("cpu"))

            if self._denormalize_input:
                x = self._denormalize(x)
            if self._denormalize_target:
                y = self._denormalize(y)
                y_pred = self._denormalize(y_pred)

            output = self._output_transform(x, y, y_pred)

            batch_sizes = [len(e) for e in output]
            assert all(s == batch_sizes[0] for s in batch_sizes)
            for i in range(batch_sizes[0]):
                grid = self._combine_test_images([e[i] for e in output])
                images.append(grid)

        wandb_images = [wandb.Image(to_pil_image(image)) for image in images]
        pipeline.log({self._log_label: wandb_images})

    def _combine_test_images(
        self, images: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        for image in images:
            assert len(image.shape) == 3
            assert image.size(0) in (1, 3)

        max_h = max(image.size(1) for image in images)
        max_w = max(image.size(2) for image in images)

        transformed = []
        for image in images:
            c, h, w = image.shape
            if (h, w) != (max_h, max_w):
                if self._resize_to_fit:
                    image = resize(
                        image,
                        size=(max_h, max_w),
                        interpolation=self._interpolation,
                    )
                else:
                    image = center_crop(image, output_size=(max_h, max_w))
            if c == 1:
                image = image.repeat((3, 1, 1))
            image = image.clamp(0.0, 1.0)
            transformed.append(image)

        return make_grid(
            tensor=transformed,
            normalize=False,
            nrow=self._num_cols or len(transformed),
        )
