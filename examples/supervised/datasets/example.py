from typing import Tuple
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import pil_to_tensor, resize


class ExampleDataset(Dataset):
    def __init__(
        self,
        split: str,
        do_augment: bool = False,
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

        self.do_augment = do_augment

    def __len__(self) -> int:
        return len(self.data)

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, _ = self.data[idx]
        image = pil_to_tensor(image) / 255

        x = resize(image, size=[16, 16], antialias=True)
        y = image

        if self.do_augment:
            x = self.augment(x)

        return x, y
