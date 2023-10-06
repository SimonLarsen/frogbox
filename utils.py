from os import PathLike
import json
from importlib import import_module
from pathlib import Path
import torch
from torchvision.transforms.functional import (
    resize,
    center_crop,
    InterpolationMode,
)
from torchvision.utils import make_grid
from ignite.engine import Events, _prepare_batch
from ignite.utils import convert_tensor
from ignite.handlers import (
    CosineAnnealingScheduler,
    create_lr_scheduler_with_warmup,
)
import jinja2
from typing import (
    Any,
    Iterable,
    Optional,
    Union,
    Dict,
    Tuple,
    Callable,
    Sequence,
    List,
)


def read_json_config(path: Union[str, PathLike]) -> Dict[str, Any]:
    """
    Read and render JSON config file and render using jinja2.
    """
    config_dir = Path(path).parent
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(config_dir)))
    template = env.get_template(str(path.relative_to(config_dir)))
    config = json.loads(template.render())
    return config


def parse_log_interval(s: Union[str, Dict[str, Any]]) -> Events:
    """
    Create ignite event from string or dictionary configuration.
    Dictionary must have a ``event`` entry.

    Example
    -------
        log_interval = {"event": "ITERATION_COMPLETED", "every": 100}
        event = parse_log_interval(log_interval)
    """
    if isinstance(s, str):
        return Events[s]
    config = dict(s)
    event = Events[config.pop("event")]
    if len(config) > 0:
        event = event(**config)
    return event


def get_class(path: str) -> Any:
    """
    Get class from import path.

    Example
    -------
        cl = get_class("torch.optim.Adam)
        optimizer = cl(lr=1e-5)
    """
    parts = path.split(".")
    module_path = ".".join(parts[:-1])
    class_name = parts[-1]
    module = import_module(module_path)
    return getattr(module, class_name)


def create_object_from_config(config: Dict[str, Any], **kwargs) -> Any:
    """
    Create object from dictionary configuration.
    Dictionary should have a ``class`` entry and an optional ``params`` entry.

    Example
    -------
        config = {"class": "torch.optim.Adam", "params": {"lr": 1e-5}}
        optimizer = create_object_from_config(config)
    """
    obj_class = get_class(config["class"])
    params = dict(config.get("params", {}))
    params.update(kwargs)
    return obj_class(**params)


def create_lr_scheduler_from_config(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    max_iterations: int,
) -> Any:
    """
    Create a learning rate scheduler from dictionary configuration.
    """
    if config["type"].lower() != "cosine":
        raise ValueError(f"Unsupported annealing schedule '{config['type']}'.")

    lr_scheduler = CosineAnnealingScheduler(
        optimizer=optimizer,
        param_name="lr",
        start_value=config["start_value"],
        end_value=config["end_value"],
        cycle_size=max_iterations,
    )

    warmup_steps = config.get("warmup_steps", 0)
    if warmup_steps > 0:
        lr_scheduler = create_lr_scheduler_with_warmup(
            lr_scheduler=lr_scheduler,
            warmup_start_value=0.0,
            warmup_end_value=config["start_value"],
            warmup_duration=warmup_steps,
        )

    return lr_scheduler


def load_model_checkpoint(
    path: Union[str, PathLike]
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load model from checkpoint."""
    path = Path(path)

    config_path = path.parent / "config.json"
    with open(config_path, "r") as fp:
        config = json.load(fp)

    model = create_object_from_config(config["model"])
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
    num_cols: int = None,
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
    num_cols: int = None,
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
