import torch
from torch import Tensor, nn
from torchvision.transforms.functional import to_pil_image

from frogbox import SupervisedPipeline


def forward(
    model: nn.Module,
    x: Tensor,
    p_t_zero: float = 0.1,
) -> tuple[Tensor, Tensor]:
    # Normalize target to [-1, 1]
    x = x * 2 - 1

    # Sample random time steps
    t = torch.rand(x.size(0), device=x.device, dtype=x.dtype)

    # Randomly set t = 0
    t = t * (torch.rand_like(t) >= p_t_zero)

    # Generate noisy sample
    noise = torch.randn_like(x)
    t_full = t.reshape(-1, 1, 1, 1)
    z = t_full * x + (1 - t_full) * noise

    # Compute ground truth velocity
    v = x - noise

    # Predict velocity from model
    v_pred = model(z, t)

    return v_pred, v


def log_images(
    pipe: SupervisedPipeline,
    num_timesteps: int = 20,
    num_images: int = 8,
    seed: int = 1234,
) -> None:
    device = pipe.device
    model = pipe.model
    model.eval()

    timesteps = torch.linspace(0, 1, num_timesteps + 1, device=device)
    dt = 1 / num_timesteps

    generator = torch.Generator(pipe.device).manual_seed(seed)

    images = []
    for _ in range(num_images):
        z = torch.randn(1, 3, 32, 32, device=device, generator=generator)

        with torch.inference_mode():
            for t in timesteps[:-1, None]:
                v = model(z, t)
                z = z + v * dt

        image = to_pil_image((z[0] / 2 + 0.5).clamp(0, 1))
        images.append(image)

    pipe.log_images("images", images)
