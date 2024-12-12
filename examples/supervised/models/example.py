import math
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        norm_groups: int = 4,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.act = nn.GELU()

        self.norm1 = nn.GroupNorm(norm_groups, channels, norm_eps)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)

        self.norm2 = nn.GroupNorm(norm_groups, channels, norm_eps)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        return x + identity


class ExampleModel(nn.Module):
    def __init__(
        self,
        scale_factor: int = 2,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_channels: int = 32,
        num_layers: int = 4,
        norm_groups: int = 4,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                ResidualBlock(
                    channels=hidden_channels,
                    norm_groups=norm_groups,
                )
            )

        upsample = []
        for _ in range(int(math.log2(scale_factor))):
            upsample.append(nn.Upsample(scale_factor=2, mode="nearest"))
            upsample.append(
                nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1)
            )
            upsample.append(nn.GELU())
        self.upsample = nn.Sequential(*upsample)

        self.conv_out = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        h = self.conv_in(x)
        for block in self.blocks:
            h = block(h) + h
        h = self.upsample(h)
        h = self.conv_out(h)

        x = nn.functional.interpolate(x, h.shape[-2:], mode="bilinear")
        return nn.functional.sigmoid(x + h)
