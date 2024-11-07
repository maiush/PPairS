from PPairS.constants import results_path, collated_results_path
from PPairS.utils import dataset_aspects, models

import os
from pathlib import Path

import pandas as pd
import torch as t
import torch.nn.functional as F


scoring_datasets = ['newsroom', 'summeval', 'hanna']
comparison_datasets = ['rocstories']
grounding_datasets = ['caters', 'mctaco']

def pc_results(dataset: str) -> None:
    if dataset in scoring_datasets or dataset in comparison_datasets:
        pc_aspect(dataset)
    elif dataset in grounding_datasets:
        pc_grounding(dataset)

def pc_grounding(dataset: str) -> None:
    data = pd.DataFrame(columns=['prob'])
    for model in models.keys():
        outpath = f"{collated_results_path}/{dataset}/{model}"
        Path(outpath).mkdir(exist_ok=True, parents=True)
        outpath += f"/compare.jsonl"
        if os.path.exists(outpath): continue
        failed = False
        pc_path = f"{results_path}/{dataset}/{model}/compare"
        pc = t.load(f"{pc_path}.pt", weights_only=True)
        pc_r = t.load(f"{pc_path}_reversed.pt", weights_only=True)
        # logits to probs
        pc, pc_r = F.softmax(pc, dim=-1), F.softmax(pc_r, dim=-1)
        # calibrate
        probs = (pc + (1 - pc_r)) / 2
        # we want an actual preference probability
        # we just store P(choice == 1)
        data['prob'] = probs[:, 0].float()
        if not failed: data.to_json(outpath, orient="records", lines=True)

def pc_aspect(dataset: str) -> None:
    aspects = dataset_aspects[dataset]
    data = pd.DataFrame(columns=aspects)
    for model in models.keys():
        outpath = f"{collated_results_path}/{dataset}/{model}"
        Path(outpath).mkdir(exist_ok=True, parents=True)
        outpath += f"/compare.jsonl"
        if os.path.exists(outpath): continue
        failed = False
        for aspect in aspects:
            pc_path = f"{results_path}/{dataset}/{model}/{aspect}_compare"
            pc = t.load(f"{pc_path}.pt", weights_only=True)
            pc_r = t.load(f"{pc_path}_reversed.pt", weights_only=True)
            # logits to probs
            pc, pc_r = F.softmax(pc, dim=-1), F.softmax(pc_r, dim=-1)
            # calibrate
            probs = (pc + (1 - pc_r)) / 2
            # we want an actual preference probability
            # we just store P(choice == 1)
            data[aspect] = probs[:, 0].float()
        if not failed: data.to_json(outpath, orient="records", lines=True)


if __name__ == '__main__':
    for dataset in ["newsroom", "summeval", "hanna", "rocstories", "caters", "mctaco"]:
        pc_results(dataset)