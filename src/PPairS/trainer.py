from PPairS.datasets import *
from torch.utils.data import DataLoader
from typing import Any, Tuple


class Trainer():

    def __init__(
        self,
        dataset_class: Any,
        dataset_kwargs: dict={},
        splits: Tuple[float]=(0.8, 0.9),
        batch_size: int=32
    ):
        self.dataset = dataset_class(**dataset_kwargs)
        # train/val/test split
        self.train_dataset, self.val_dataset, self.test_dataset = self.dataset.get_splits(splits)
        # dataloaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)