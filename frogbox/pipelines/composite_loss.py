import inspect
from typing import Sequence
import torch


class CompositeLoss(torch.nn.Module):
    """
    Criterion that is a weighted sum of multiple loss functions.
    """

    def __init__(
        self,
        labels: Sequence[str],
        losses: Sequence[torch.nn.Module],
        weights: Sequence[float],
    ):
        super().__init__()

        assert len(labels) == len(losses) == len(weights)

        self.labels = labels
        self.losses = torch.nn.ModuleList(losses)
        self.weights = weights
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
        for i, (w, l) in enumerate(zip(self.weights, self.losses)):
            extra_args = self._gather_extra_args(l, kwargs)
            loss = w * l(input, target, **extra_args)
            total_loss += loss
            self.last_values[i] = loss.item()
        return total_loss
