import math
from torch import nn


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name in ("swish", "silu"):
        return nn.SiLU()
    elif name == "mish":
        return nn.Mish()
    else:
        raise ValueError(f"Unsupported activation function '{name}'.")


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        norm_groups: int = 4,
        norm_eps: float = 1e-5,
        activation: str = "relu",
    ):
        super().__init__()

        self.act = get_activation(activation)

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
        activation="gelu",
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)

        features = []
        for _ in range(num_layers):
            features.append(
                ResidualBlock(
                    channels=hidden_channels,
                    norm_groups=norm_groups,
                    activation=activation,
                )
            )
        features.append(nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1))
        self.features = nn.Sequential(*features)

        upsample = []
        for _ in range(int(math.log2(scale_factor))):
            upsample.append(nn.Upsample(scale_factor=2, mode="nearest"))
            upsample.append(
                nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1)
            )
            upsample.append(get_activation(activation))
        self.upsample = nn.Sequential(*upsample)

        self.conv_out = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            get_activation(activation),
            nn.Conv2d(hidden_channels, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.features(x)
        x = self.upsample(x)
        x = self.conv_out(x)
        return x
