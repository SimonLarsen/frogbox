from typing import cast, Any, Callable
from collections.abc import Mapping, Sequence
import torch


def apply_to_tensor(x: Any, function: Callable) -> Any:
    """
    Recursively apply `function` to all tensors in collection.

    Parameters
    ----------
    x : tensor or collection containing tensors
        Object to apply function to.
    function : callable
        Function that takes a single tensors as argument.

    Returns
    -------
    A new collection of same type as `x`.
    """

    if isinstance(x, torch.Tensor):
        return function(x)
    if isinstance(x, (str, bytes)):
        return x
    if isinstance(x, Mapping):
        return cast(Callable, type(x))(
            {k: apply_to_tensor(sample, function) for k, sample in x.items()}
        )
    if isinstance(x, tuple) and hasattr(x, "_fields"):  # namedtuple
        return cast(Callable, type(x))(
            *(apply_to_tensor(sample, function) for sample in x)
        )
    if isinstance(x, Sequence):
        return cast(Callable, type(x))(
            [apply_to_tensor(sample, function) for sample in x]
        )
    raise TypeError(
        (f"x must contain tensors, dicts or lists; found {type(x)}")
    )


def convert_tensor(
    x: Any,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Any:
    """
    Recursively convert tensors in collection to `dtype` and/or move to
    `device`.

    Parameters
    ----------
    device : torch.device
        CUDA device to move tensors to.
    dtype : torch.dtype
        dtype to convert tensors to.
    """

    def _convert(e: torch.Tensor) -> torch.Tensor:
        return e.to(device=device, dtype=dtype)

    return apply_to_tensor(x, _convert)
