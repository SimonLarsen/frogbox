from torch import nn
from torch.nn.utils import spectral_norm
from .blocks import get_activation


class SNResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        activation: str = "relu",
    ):
        super().__init__()

        self.act = get_activation(activation)
        self.conv1 = spectral_norm(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        )
        self.conv2 = spectral_norm(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        )

    def forward(self, x):
        identity = x

        x = self.act(x)
        x = self.conv1(x)

        x = self.act(x)
        x = self.conv2(x)

        return x + identity


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 32,
        num_blocks: int = 8,
        activation: str = "silu",
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(SNResidualBlock(hidden_channels, activation))
        self.blocks = nn.Sequential(*blocks)

        self.conv_out = nn.Conv2d(hidden_channels, 1, 3, 1, 1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.conv_out(x)
        return x
