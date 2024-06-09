from PPairS.datasets import *
from torch.utils.data import Subset, DataLoader
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
        trainval_split, valtest_split = int(splits[0]*len(self.dataset)), int(splits[1]*len(self.dataset))
        # shuffle indices and create subsets
        perm = t.randperm(len(self.dataset))
        self.train_dataset = Subset(self.dataset, perm[:trainval_split])
        self.val_dataset = Subset(self.dataset, perm[trainval_split:valtest_split])
        self.test_dataset = Subset(self.dataset, perm[valtest_split:])
        # dataloaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)