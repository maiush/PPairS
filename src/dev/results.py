from PPairS.constants import home_path, data_path, results_path, collated_results_path
from PPairS.utils import dataset_aspects, models

import os
from pathlib import Path

import pandas as pd
import torch as t
import torch.nn.functional as F


# direct scoring
for dataset in ["newsroom", "summeval", "hanna"]:
    data = pd.read_json(f"{data_path}/{dataset}.jsonl", orient="records", lines=True)
    aspects = dataset_aspects[dataset]
    data = data[aspects]
    for model in models.keys():
        outpath = f"{collated_results_path}/{dataset}/{model}"
        Path(outpath).mkdir(exist_ok=True, parents=True)
        outpath += f"/score.jsonl"
        if os.path.exists(outpath): continue
        for aspect in aspects:
            # n_data, n_seq, n_logit (n_score)
            zs = t.load(f"{results_path}/{dataset}/{model}/{aspect}_zero_shot.pt", weights_only=True)
            # calculate the total probability of each possible score
            # this is sum(P(x_i == score) for all i)
            zs = F.softmax(zs, dim=-1)
            # add one to convert to numerical score
            zs = t.nan_to_num(zs).sum(dim=1).argmax(dim=-1) + 1
            data[aspect] = zs
        data.to_json(outpath, orient="records", lines=True)

# g-eval: https://arxiv.org/pdf/2303.16634
for dataset in ["newsroom", "summeval", "hanna"]:
    data = pd.read_json(f"{data_path}/{dataset}.jsonl", orient="records", lines=True)
    aspects = dataset_aspects[dataset]
    data = data[aspects]
    for model in models.keys():
        outpath = f"{collated_results_path}/{dataset}/{model}"
        Path(outpath).mkdir(exist_ok=True, parents=True)
        outpath += f"/geval.jsonl"
        if os.path.exists(outpath): continue
        for aspect in aspects:
            # n_data, n_seq, n_logit (n_score)
            zs = t.load(f"{results_path}/{dataset}/{model}/{aspect}_zero_shot.pt", weights_only=True)
            # calculate the total probability of each possible score
            # this is sum(P(x_i == score) for all i)
            zs = F.softmax(zs, dim=-1)
            zs = t.nan_to_num(zs).sum(dim=1)
            # normalise
            zs = F.softmax(zs, dim=-1)
            # g-eval method (weighted average of possible scores)
            zs = zs * t.arange(start=1, end=6, dtype=zs.dtype, device=zs.device)[None, :].repeat(zs.shape[0], 1)
            zs = t.round(zs.sum(dim=1))
            data[aspect] = zs
        data.to_json(outpath, orient="records", lines=True)

# pairwise comparisons
for dataset in ["newsroom", "summeval", "hanna"]:
    aspects = dataset_aspects[dataset]
    data = pd.read_json(f"{data_path}/{dataset}_pairwise_comparisons.jsonl", orient="records", lines=True)
    data = data[aspects]
    for model in models.keys():
        outpath = f"{collated_results_path}/{dataset}/{model}"
        Path(outpath).mkdir(exist_ok=True, parents=True)
        outpath += f"/compare.jsonl"
        if os.path.exists(outpath): continue
        for aspect in aspects:
            # n_data, n_seq, n_logit (n_choice)
            pc = t.load(f"{results_path}/{dataset}/{model}/{aspect}_compare.pt", weights_only=True)
            # as with direct scoring, we calculate the cumulative probability over the full generation
            pc = F.softmax(pc, dim=-1)
            pc = t.nan_to_num(pc).sum(dim=1) 
            # we want an actual preference probability, so we normalise
            # we just store P(choice == 1)
            pc = (pc / pc.sum(dim=-1, keepdim=True))[:, 0]
            data[aspect] = pc.float()
        data.to_json(outpath, orient="records", lines=True)