from typing import Optional, Sequence, Callable, Any, TypeAlias
import inspect
import torch


LossTransform: TypeAlias = Callable[[Any, Any], Any]


class CompositeLoss(torch.nn.Module):
    """
    Criterion that is a weighted sum of multiple loss functions.
    """

    def __init__(
        self,
        labels: Sequence[str],
        losses: Sequence[torch.nn.Module],
        weights: Sequence[float],
        transforms: Optional[Sequence[Optional[LossTransform]]] = None,
    ):
        super().__init__()

        assert len(labels) == len(losses) == len(weights)
        if transforms is None:
            transforms = [None] * len(labels)
        assert len(transforms) == len(labels)

        self.labels = labels
        self.losses = torch.nn.ModuleList(losses)
        self.weights = weights
        self.transforms = transforms
        self.last_values = [None] * len(losses)

    def _gather_extra_args(self, loss_fn: torch.nn.Module, kwargs):
        sig = inspect.signature(loss_fn.forward)
        args = sig.parameters.keys()
        return {k: v for k, v in kwargs.items() if k in args}

    def forward(self, input, target, **kwargs):
        """
        Compute loss.
        """
        total_loss = 0.0
        for i, (weight, loss_fn, transform) in enumerate(
            zip(self.weights, self.losses, self.transforms)
        ):
            args = (input, target)
            if transform is not None:
                args = transform(*args)
            extra_args = self._gather_extra_args(loss_fn, kwargs)
            loss = weight * loss_fn(*args, **extra_args)
            total_loss += loss
            self.last_values[i] = loss.item()
        return total_loss
