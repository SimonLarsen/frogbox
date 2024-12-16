import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import pil_to_tensor, resize
from typing import Tuple, Sequence


NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)


class ExampleDataset(Dataset):
    def __init__(
        self,
        split: str,
        do_normalize: bool = True,
        do_augment: bool = False,
        normalize_mean: Sequence[float] = NORMALIZE_MEAN,
        normalize_std: Sequence[float] = NORMALIZE_STD,
        download: bool = True,
    ):
        super().__init__()

        split = split.lower()
        assert split in ("train", "val", "test")
        self.data = CIFAR10(
            root="data", train=split == "train", download=download
        )

        if split == "val":
            self.data.data = self.data.data[:-32]
        elif split == "test":
            self.data.data = self.data.data[-32:]

        self.do_normalize = do_normalize
        self.do_augment = do_augment
        self.normalize_mean = torch.tensor(normalize_mean).reshape((3, 1, 1))
        self.normalize_std = torch.tensor(normalize_std).reshape((3, 1, 1))

    def __len__(self) -> int:
        return len(self.data)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(
            self.normalize_mean, dtype=x.dtype, device=x.device
        )
        std = torch.as_tensor(
            self.normalize_std, dtype=x.dtype, device=x.device
        )
        return (x - mean) / std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(
            self.normalize_mean, dtype=x.dtype, device=x.device
        )
        std = torch.as_tensor(
            self.normalize_std, dtype=x.dtype, device=x.device
        )
        return (x * std) + mean

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, _ = self.data[idx]
        image = pil_to_tensor(image) / 255

        x = resize(image, size=16, antialias=True)
        y = image

        if self.do_augment:
            x = self.augment(x)

        if self.do_normalize:
            x = self.normalize(x)

        return x, y
