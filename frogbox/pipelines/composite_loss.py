import torch
from typing import Sequence


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

    def forward(self, input, target):
        """
        Compute loss.
        """
        total_loss = 0.0
        for i, (w, l) in enumerate(zip(self.weights, self.losses)):
            loss = w * l(input, target)
            total_loss += loss
            self.last_values[i] = loss.item()
        return total_loss
