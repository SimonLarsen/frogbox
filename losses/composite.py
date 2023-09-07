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

    def forward(self, input, target):
        loss = 0.0
        for w, l in zip(self.weights, self.losses):
            loss += w * l(input, target)
        return loss
