from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import torch

LossTransform: TypeAlias = Callable[..., Any]


class CompositeLoss(torch.nn.Module):
    """
    Criterion that is a weighted sum of multiple loss functions.
    """

    def __init__(
        self,
        labels: Sequence[str],
        losses: Sequence[torch.nn.Module],
        weights: Sequence[float],
        transforms: Sequence[LossTransform | None] | None = None,
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

    def forward(self, *args):
        """
        Compute loss.
        """
        total_loss = 0.0
        for i, (weight, loss_fn, transform) in enumerate(
            zip(self.weights, self.losses, self.transforms)
        ):
            if transform is not None:
                args = transform(*args)
            loss = weight * loss_fn(*args)
            total_loss += loss
            self.last_values[i] = loss.item()
        return total_loss
