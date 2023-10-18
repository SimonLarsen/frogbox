from torch.utils.data import Dataset


class ExampleDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int):
        pass
