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
