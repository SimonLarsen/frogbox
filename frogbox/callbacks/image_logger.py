from typing import Sequence, Optional, Callable, Any, Tuple
import torch
from torchvision.transforms.functional import (
    center_crop,
    resize,
    InterpolationMode,
    to_pil_image,
)
from torchvision.utils import make_grid
import tqdm
from .callback import Callback
from ..pipelines.pipeline import Pipeline
from ..tensor_utils import convert_tensor


def _default_forward(x: Any, y: Any, model: Callable) -> Tuple[Any, ...]:
    return x, model(x), y


class ImageLogger(Callback):
    """Callback for logging images."""

    def __init__(
        self,
        split: str = "test",
        log_label: str = "images",
        model_key: str = "model",
        resize_to_fit: bool = True,
        interpolation: str | InterpolationMode = "nearest",
        num_cols: Optional[int] = None,
        show_progress: bool = False,
        forward: Optional[
            Callable[[Any, Any, Callable], Tuple[Any, ...]]
        ] = None,
    ):
        """
        Create ImageLogger.

        Parameters
        ----------
        split : str
            Dataset split to evaluate on. Defaults to "test".
        log_label : str
            Label to log images under in Weights & Biases.
        model_key : str
            Pipeline model to use for inference.
        resize_to_fit : bool
            If `true` smaller images are resized to fit canvas.
        interpolation : torchvision.transforms.functional.InterpolationMode
            Interpolation to use for resizing images.
        forward : callable
            Function that arguments `x`, `y` and `model` and tuple of images
            to log. Returns `(x, model(x), y)` if not provided.
        show_progress : bool
            Show progress bar.
        num_cols : int
            Number of columns in image grid.
            Defaults to number of elements in returned tuple.
        """
        if forward is None:
            forward = _default_forward

        self.split = split
        self.log_label = log_label
        self.model_key = model_key
        self.resize_to_fit = resize_to_fit
        self.interpolation = InterpolationMode(interpolation)
        self.forward = forward
        self.show_progress = show_progress
        self.num_cols = num_cols

    def __call__(self, pipeline: Pipeline) -> None:
        model = pipeline._models[self.model_key]
        loader = pipeline._loaders[self.split]
        accelerator = pipeline.accelerator

        model.eval()

        data_iter = loader
        if self.show_progress:
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

            with torch.inference_mode():
                outputs = self.forward(x, y, model)

            outputs = accelerator.gather_for_metrics(outputs)

            outputs = tuple(
                convert_tensor(e, torch.device("cpu")) for e in outputs
            )
            batch_sizes = [len(e) for e in outputs]
            assert all(s == batch_sizes[0] for s in batch_sizes)
            for i in range(batch_sizes[0]):
                grid = self._combine_test_images([e[i] for e in outputs])
                images.append(grid)

        pil_images = [to_pil_image(image) for image in images]
        pipeline.log_images(self.log_label, pil_images)

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
                if self.resize_to_fit:
                    image = resize(
                        image,
                        size=[max_h, max_w],
                        interpolation=self.interpolation,
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
            nrow=self.num_cols or len(transformed),
        )
