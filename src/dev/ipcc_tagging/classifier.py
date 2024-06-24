import os, dill
from dev.constants import gdrive_path

import torch as t
from torch import nn, Tensor
from torch.utils.data import Dataset, Subset, DataLoader
import pandas as pd

from functools import reduce
from random import shuffle, choices
from jaxtyping import Float
from typing import Any, Optional, List, Tuple
from tqdm.notebook import tqdm


class IPCCDataset(Dataset):

    def __init__(
            self,
            subset_reports: Optional[List[str]]=["AR3", "AR4", "AR5", "AR6"],
            subset_parts: Optional[List[int]]=None,
            device: str="cuda"
    ):
        self.device = t.device(device)

        # all prompts used: 1-1 correspondence with activations
        self.files = os.listdir(f"{gdrive_path}/ipcc_tagging/prompts")
        if subset_reports is not None:
            self.files = [f for report in subset_reports for f in self.files if report in f]
        if subset_parts is not None:
            self.files = [f for part in subset_parts for f in self.files if f"_PART{part}.jsonl" in f]

        # we index over all harvested activations as one dataset
        self.file_ranges = []
        # we will cache all activations and labels for faster data loading
        self.act_cache, self.labels_cache, self.id_cache, ix = {}, {}, {}, 0
        for file in tqdm(self.files, desc="caching data"):
            data = t.load(f"{gdrive_path}/ipcc_tagging/activations/{file.replace('jsonl', 'pt')}", pickle_module=dill)
            prompts = pd.read_json(f"{gdrive_path}/ipcc_tagging/prompts/{file}", orient="records", lines=True)
            ids = prompts.apply(lambda row: (row["S1"], row["S2"]), axis=1).to_numpy()
            labels = t.from_numpy(prompts["P(S1)"].to_numpy())

            self.act_cache[file] = data.to(self.device)
            self.labels_cache[file] = labels.to(data.dtype).to(self.device)
            self.id_cache[file] = ids

            N = data.size(0)
            self.file_ranges.append((ix, ix+N))
            ix += N
    
    def __len__(self):
        return self.file_ranges[-1][1]
    
    def __getitem__(self, ix):
        file_ix = next(i for i, (start, end) in enumerate(self.file_ranges) if start <= ix < end)
        file = self.files[file_ix]
        data_ix = ix - self.file_ranges[file_ix][0]
        
        data = self.act_cache[file][data_ix]
        label = self.labels_cache[file][data_ix]
        return data, label

    def get_splits(self, splits: Tuple[float]=(0.8, 0.9)) -> Tuple[Subset]:
        all_train_ixs, all_val_ixs, all_test_ixs = [], [], []
        # for each report
        for i in range(1, 7):
            report = f"AR{i}"

            # split the IDs into train/val/test sets
            files = [filename for filename in self.files if report in filename]
            if len(files) == 0: continue
            ids = [self.id_cache[filename] for filename in files]
            ids = map(lambda ids: set([pair[0] for pair in ids]), ids)
            ids = list(reduce(lambda x, y: x.union(y), ids))
            shuffle(ids)
            trainval_split, valtest_split = int(splits[0]*len(ids)), int(splits[1]*len(ids))
            train_ids = ids[:trainval_split]
            val_ids = ids[trainval_split:valtest_split]
            test_ids = ids[valtest_split:]

            # find the corresponding data indices for the above IDs
            train_ixs, val_ixs, test_ixs = [], [], []
            for filename in files:
                file_ix = self.files.index(filename)
                start, _ = self.file_ranges[file_ix]
                train_ixs.extend([start+i for i, x in enumerate(self.id_cache[filename]) if x[0] in train_ids])
                val_ixs.extend([start+i for i, x in enumerate(self.id_cache[filename]) if x[0] in val_ids])
                test_ixs.extend([start+i for i, x in enumerate(self.id_cache[filename]) if x[0] in test_ids])

            all_train_ixs.append(train_ixs)
            all_val_ixs.append(val_ixs)
            all_test_ixs.append(test_ixs)

        # resample from under-sampled reports
        for group in [all_train_ixs, all_val_ixs, all_test_ixs]:
            N = max([len(x) for x in group])
            for i in range(len(group)):
                ixs = group[i]
                delta = N - len(ixs)
                if delta > 0: 
                    group[i] = ixs + choices(ixs, k=delta)

        # create subsets for each dataset
        all_train_ixs = [ix for ixs in all_train_ixs for ix in ixs]
        all_val_ixs = [ix for ixs in all_val_ixs for ix in ixs]
        all_test_ixs = [ix for ixs in all_test_ixs for ix in ixs]
        train_dataset = Subset(self, all_train_ixs)
        val_dataset = Subset(self, all_val_ixs)
        test_dataset = Subset(self, all_test_ixs)

        return (train_dataset, val_dataset, test_dataset)


class Trainer():

    def __init__(
        self,
        dataset_class: Any,
        dataset_kwargs: dict={},
        splits: Tuple[float]=(0.8, 0.9),
        batch_size: int=64
    ):
        # dataset and subsets
        self.dataset = dataset_class(**dataset_kwargs)
        self.train_dataset, self.val_dataset, self.test_dataset = self.dataset.get_splits(splits)
        # dataloaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)


class PPairSClassifier(nn.Module):

    def __init__(
        self,
        d_model: int=4096,
        d_out: int=1,
        dtype: t.dtype=t.float32,
        device: str="cuda"
    ):
        super(PPairSClassifier, self).__init__()
        self.d_model = d_model
        self.d_out = d_out
        self.dtype = dtype
        self.device = t.device(device)

        self.probe = nn.Linear(
            in_features=self.d_model,
            out_features=self.d_out,
            bias=True,
            dtype=self.dtype,
            device=self.device
        )

    def forward(
        self,
        x: Float[Tensor, "batch d_model"]
    ) -> Float[Tensor, "d_out"]:
        return self.probe(x)


trainer = Trainer(
    dataset_class=IPCCDataset,
    dataset_kwargs={
        "subset_reports": ["AR3", "AR4", "AR5", "AR6"],
        "subset_parts": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "device": "cuda"
    },
    splits=(0.8, 0.9),
    batch_size=64
)

# calculate weights for classification
totalZ, totalH, totalO = 0, 0, 0
for batch, labels in trainer.train_loader:
    totalZ += (labels == 0).sum()
    totalH += (labels == 0.5).sum()
    totalO += (labels == 1).sum()
total = totalZ + totalH + totalO
w = t.stack([(totalH + totalO) / totalZ, (totalZ + totalO) / totalH, (totalZ + totalH) / totalO])


d_out = int(sys.argv[1])
if d_out == 1:
    clfs = [PPairSClassifier(dtype=t.float32, d_out=1) for _ in range(3)]
    opts = [t.optim.Adam(clf.parameters(), lr=0.01, weight_decay=0.) for clf in clfs]
    for epoch in range(n_epoch):
        for batch, labels in trainer.train_loader:
            y = [
                (labels == 0.),
                (labels == 0.5),
                (labels == 1.)
            ]
            for i in range(3):
                logits = clfs[i](batch)
                loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), y[i], pos_weight=w[i])
                opts[i].zero_grad()
                loss.backward()
                opts[i].step()

        accuracy = []
        for batch, labels in trainer.val_loader:
            class_ixs = (labels * 2.)
            with t.inference_mode(): logits = t.stack([clf(batch) for clf in clfs]).squeeze(-1)
            predictions = logits.argmax(dim=0)
            accuracy.append((predictions == class_ixs).float().mean())
        accuracy = sum(accuracy) / len(accuracy)
        print(f"epoch {epoch+1}: {round(accuracy.item(), 3)}")
        
        if accuracy > 0.8: break

    accuracy = []
    for batch, labels in trainer.test_loader:
        class_ixs = (labels * 2.)
        with t.inference_mode(): logits = t.stack([clf(batch) for clf in clfs]).squeeze(-1)
        predictions = logits.argmax(dim=0)
        accuracy.append((predictions == class_ixs).float().mean())
    accuracy = sum(accuracy) / len(accuracy)
    print("-"*50)
    print(f"test accuracy: {round(accuracy.item(), 3)}")

    path = "/gws/nopw/j04/ai4er/users/maiush/PPairS/src/dev/_ipcc_tagging"
    t.save(clfs[0].state_dict(), f"{path}/clf_zero.pt", pickle_module=dill)
    t.save(clfs[1].state_dict(), f"{path}/clf_half.pt", pickle_module=dill)
    t.save(clfs[2].state_dict(), f"{path}/clf_one.pt", pickle_module=dill)

elif d_out == 3:

    clf = PPairSClassifier(dtype=t.float32, d_out=3)
    opt = t.optim.Adam(clf.parameters(), lr=0.01, weight_decay=0.)
    for epoch in range(n_epoch):
        for batch, labels in trainer.train_loader:
            logits = clf(batch)
            class_ixs = (labels * 2.).to(int)
            loss = F.cross_entropy(logits, class_ixs, weight=w)

            opt.zero_grad()
            loss.backward()
            opt.step()

        accuracy = []
        for batch, labels in trainer.val_loader:
            with t.inference_mode(): logits = clf(batch)
            class_ixs = (labels * 2.).to(int)
            accuracy.append((logits.argmax(dim=-1) == class_ixs).float().mean())
        accuracy = sum(accuracy) / len(accuracy)
        print(f"epoch {epoch+1}: {round(accuracy.item(), 3)}")

        if accuracy > 0.8: break

    accuracy = []
    for batch, labels in trainer.test_loader:
        with t.inference_mode(): logits = clf(batch)
        class_ixs = (labels * 2.).to(int)
        accuracy.append((logits.argmax(dim=-1) == class_ixs).float().mean())
    accuracy = sum(accuracy) / len(accuracy)
    print("-"*50)
    print(f"test accuracy: {round(accuracy.item(), 3)}")

    path = "/gws/nopw/j04/ai4er/users/maiush/PPairS/src/dev/_ipcc_tagging"
    t.save(clf.state_dict(), f"{path}/clf_multi.pt", pickle_module=dill)