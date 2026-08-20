from torch import Tensor, nn


class LayerNorm2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class ResNetConditionalBlock(nn.Module):
    def __init__(self, channels: int, condition_dim: int):
        super().__init__()

        self.act = nn.GELU()

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)

        self.norm2 = LayerNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

        self.cond_proj = nn.Linear(condition_dim, 2 * channels)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        scale, shift = self.cond_proj(c)[..., None, None].chunk(2, dim=1)

        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        h = self.norm2(h) * (1 + scale) + shift
        h = self.act(h)
        h = self.conv2(h)

        return x + h


class FlowMatchingModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        time_embed_dim: int = 32,
        num_layers: int = 10,
    ):
        super().__init__()

        self.time_embed = nn.Linear(1, time_embed_dim)

        self.proj_in = nn.Conv2d(3, hidden_dim, 3, 1, 1)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(ResNetConditionalBlock(hidden_dim, time_embed_dim))

        self.proj_out = nn.Conv2d(hidden_dim, 3, 3, 1, 1)

    def forward(self, z: Tensor, t: Tensor) -> Tensor:
        c = self.time_embed(t.reshape(-1, 1))

        h = self.proj_in(z)

        for layer in self.layers:
            h = layer(h, c)

        h = self.proj_out(h)
        return h
