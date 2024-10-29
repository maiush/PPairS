from PPairS.constants import data_path, results_path, collated_results_path
from PPairS.utils import dataset_aspects, models

import os
from pathlib import Path

import pandas as pd
import torch as t
import torch.nn.functional as F


def ds_results(dataset: str, g_eval: bool=False) -> None:
    data = pd.read_json(f"{data_path}/{dataset}.jsonl", orient="records", lines=True)
    aspects = dataset_aspects[dataset]
    data = data[aspects]
    for model in models.keys():
        outpath = f"{collated_results_path}/{dataset}/{model}"
        Path(outpath).mkdir(exist_ok=True, parents=True)
        if g_eval: outpath += f"/geval.jsonl"
        else: outpath += f"/score.jsonl"
        if os.path.exists(outpath): continue
        for aspect in aspects:
            zs_path = f"{results_path}/{dataset}/{model}/{aspect}_zero_shot.pt"
            if not os.path.exists(zs_path):
                print(f"results {zs_path} do not exist")
                continue
            # n_data, n_seq, n_logit (n_score)
            zs = t.load(zs_path, weights_only=True)
            if g_eval:
                # logits to probs
                zs = F.softmax(zs, dim=-1)
                # g-eval method (weighted average of possible scores)
                zs = zs * t.arange(start=1, end=6, dtype=zs.dtype, device=zs.device)[None, :].repeat(zs.shape[0], 1)
                zs = t.round(zs.sum(dim=1)).int()
            else:
                # most probable answer + add one to convert to numerical score
                zs = zs.argmax(dim=-1) + 1
            data[aspect] = zs
        data.to_json(outpath, orient="records", lines=True)


if __name__ == '__main__':
    for dataset in ["newsroom", "summeval", "hanna"]:
        # zero-shot
        ds_results(dataset, False)
        # g-eval: https://arxiv.org/pdf/2303.16634
        ds_results(dataset, True)