import torch
from typing import Sequence


class CompositeLoss(torch.nn.Module):
    def __init__(
        self,
        losses: Sequence[torch.nn.Module],
        weights: Sequence[float],
    ):
        super().__init__()

        assert len(losses) == len(weights)
        self.losses = torch.nn.ModuleList(losses)
        self.weights = weights
        self.last_values = [None] * len(losses)

    def forward(self, input, target):
        total_loss = 0.0
        for i, (w, l) in enumerate(zip(self.weights, self.losses)):
            loss = w * l(input, target)
            total_loss += loss
            self.last_values[i] = loss.item()
        return total_loss
