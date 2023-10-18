from torch import nn


class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
