import os, dill
from dev.constants import gdrive_path

import pandas as pd
import torch as t
from torch.utils.data import Dataset

from typing import Optional, List
from tqdm import tqdm


class IPCCDataset(Dataset):

    def __init__(self, subset: Optional[List[str]], return_id: bool=False, contrast: bool=False, device: str="cuda"):
        self.return_id = return_id
        self.contrast = contrast
        self.device = device
        # all prompts used - corresponds to activations harvested
        self.files = os.listdir(f"{gdrive_path}/ipcc/long/prompts")
        if subset is not None:
            files = []
            for x in subset:
                files.extend([f for f in self.files if x in f])
            self.files = files
        # we index over all harvested activations as one dataset
        self.file_ranges = []
        # we will cache all activations and labels for faster loading
        self.cache, self.cache_pos, self.cache_neg, self.labels, self.ids = {}, {}, {}, {}, {}
        ix = 0
        for file in tqdm(self.files, desc="caching data"):
            labels = pd.read_json(f"{gdrive_path}/ipcc/long/prompts/{file}", orient="records", lines=True)
            path = f"{gdrive_path}/ipcc/long/activations"
            f = file.replace('prompts', 'PART')
            choice1 = t.load(f"{path}/{f.replace('.jsonl', '_CHOICE1.pt')}", pickle_module=dill).to(self.device)
            choice2 = t.load(f"{path}/{f.replace('.jsonl', '_CHOICE2.pt')}", pickle_module=dill).to(self.device)
            labels = t.from_numpy(labels["P(S1)"].values).to(choice1.dtype).to(device)
            ids = labels.apply(lambda row: [file[:3], row["S1"], row["S2"]], axis=1).tolist()
            if self.contrast:
                self.cache_pos[file] = choice1
                self.cache_neg[file] = choice2
                self.labels[file] = labels
            else:
                data = t.concat([choice1-choice2, choice2-choice1])
                labels = t.concat([labels, 1-labels])
                ids = ids + [[x, z, y] for x, y, z in ids]
                self.cache[file] = data
                self.labels[file] = labels
            if self.return_id:
                self.ids[file] = ids
            self.file_ranges.append((ix, ix+choice1.size(0)))
            ix += choice1.size(0)

    def __len__(self):
        return self.file_ranges[-1][1]

    def __getitem__(self, ix):
        file_ix = next(i for i, (start, end) in enumerate(self.file_ranges) if start <= ix < end)
        file = self.files[file_ix]
        data_ix = ix - self.file_ranges[file_ix][0]
        label = self.labels[file][data_ix]

        if self.contrast:
            data_pos = self.cache_pos[file][data_ix]
            data_neg = self.cache_neg[file][data_ix]
            out = [data_pos, data_neg, label]
        else:
            data = self.cache[file][data_ix]
            out = [data, label]
        if self.return_id:
            out += [self.ids[file][data_ix]]
        return out