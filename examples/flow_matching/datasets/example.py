import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms.v2.functional import pil_to_tensor, to_dtype


class ExampleDataset(Dataset):
    def __init__(
        self,
        split: str,
        download: bool = True,
    ):
        super().__init__()

        split = split.lower()
        assert split in ("train", "val", "test")
        self.data = CIFAR10(root="data", train=split == "train", download=download)

        if split == "val":
            self.data.data = self.data.data[:-32]
        elif split == "test":
            self.data.data = self.data.data[-32:]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        image, _ = self.data[idx]
        image = to_dtype(
            pil_to_tensor(image),
            torch.float32,
            scale=True,
        )
        return (image,)
