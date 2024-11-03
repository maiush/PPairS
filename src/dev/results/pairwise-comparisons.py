from PPairS.constants import data_path, results_path, collated_results_path
from PPairS.utils import dataset_aspects, models

import os
from pathlib import Path

import pandas as pd
import torch as t
import torch.nn.functional as F

def pc_results(dataset: str) -> None:
    aspects = dataset_aspects[dataset]
    data = pd.read_json(f"{data_path}/{dataset}_pairwise_comparisons.jsonl", orient="records", lines=True)
    data = data[aspects]
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
            probs = (pc + pc_r) / 2
            # we want an actual preference probability
            # we just store P(choice == 1)
            data[aspect] = probs[:, 0].float()
        if not failed: data.to_json(outpath, orient="records", lines=True)


if __name__ == '__main__':
    for dataset in ["newsroom", "summeval", "hanna"]:
        pc_results(dataset)

    # TODO: rocstories