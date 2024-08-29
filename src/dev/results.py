from dev.constants import home_path, data_path, results_path
from PPairS.utils import dataset_aspects, models

import os
from pathlib import Path

import pandas as pd
import torch as t
import torch.nn.functional as F


for dataset in ["newsroom", "summeval", "hanna"]:
    data = pd.read_json(f"{data_path}/{dataset}.jsonl", orient="records", lines=True)
    aspects = dataset_aspects[dataset]
    data = data[aspects]
    for model in models.keys():
        if model == "mistral-large-123b": continue
        outpath = f"{home_path}/PPairS_results/{dataset}/{model}"
        Path(outpath).mkdir(exist_ok=True, parents=True)
        outpath += f"/score.jsonl"
        if os.path.exists(outpath): continue
        for aspect in aspects:
            zs = t.load(f"{results_path}/{dataset}/{model}/{aspect}_zero_shot.pt", weights_only=True)
            zs = F.softmax(zs, dim=-1)
            zs = t.nan_to_num(zs).sum(dim=1).argmax(dim=-1) + 1
            data[aspect] = zs
        data.to_json(outpath, orient="records", lines=True)


for dataset in ["newsroom", "summeval", "hanna"]:
    aspects = dataset_aspects[dataset]
    data = pd.read_json(f"{data_path}/{dataset}_pairwise_comparisons.jsonl", orient="records", lines=True)
    data = data[aspects]
    for model in models.keys():
        if model == "mistral-large-123b": continue
        outpath = f"{home_path}/PPairS_results/{dataset}/{model}"
        Path(outpath).mkdir(exist_ok=True, parents=True)
        outpath += f"/compare.jsonl"
        if os.path.exists(outpath): continue
        for aspect in aspects:
            pc = t.load(f"{results_path}/{dataset}/{model}/{aspect}_compare.pt", weights_only=True)
            pc = F.softmax(pc, dim=-1)
            pc = t.nan_to_num(pc).sum(dim=1) 
            pc = (pc / pc.sum(dim=-1, keepdim=True))[:, 0]
            data[aspect] = pc.float()
        data.to_json(outpath, orient="records", lines=True)