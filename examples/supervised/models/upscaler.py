import math
from torch import nn
from torch.nn.functional import interpolate


class LayerNorm2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(*args, **kwargs)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.act = nn.GELU()

        self.blocks = nn.Sequential(
            LayerNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            LayerNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.blocks(x)


class Upsample(nn.Sequential):
    def __init__(self, channels: int):
        super().__init__()
        self.append(nn.Conv2d(channels, 4 * channels, 3, 1, 1))
        self.append(nn.PixelShuffle(2))


class Upscaler(nn.Module):
    def __init__(
        self,
        scale_factor: int = 2,
        hidden_channels: int = 32,
        num_layers: int = 4,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(3, hidden_channels, 3, 1, 1)

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(ResidualBlock(hidden_channels))

        self.upsample = nn.ModuleList()
        for _ in range(int(math.log2(scale_factor))):
            self.upsample.append(Upsample(hidden_channels))

        self.readout = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            LayerNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 3, 3, 1, 1),
        )

    def forward(self, x):
        h = self.conv_in(x)
        for block in self.blocks:
            h = block(h)
        for block in self.upsample:
            h = block(h)
        h = self.readout(h)
        x = interpolate(x, h.shape[-2:], mode="bilinear")
        return x + h
