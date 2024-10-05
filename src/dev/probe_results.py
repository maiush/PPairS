from PPairS.constants import collated_results_path
from PPairS.utils import models, dataset_aspects
import pandas as pd

models = list(models.keys())


all_results = []
for dataset in ["newsroom", "summeval", "hanna"]:
    aspects = dataset_aspects[dataset]
    b = pd.read_json(
        f"{collated_results_path}/{dataset}/baseline_results.jsonl",
        orient="records",
        lines=True
    )
    s = pd.read_json(
        f"{collated_results_path}/{dataset}/probe_s_results.jsonl",
        orient="records",
        lines=True
    )
    u = pd.read_json(
        f"{collated_results_path}/{dataset}/probe_u_results.jsonl",
        orient="records",
        lines=True
    )
    b = b.loc[b["method"] == "pairwise_comparisons", ["model"]+aspects]
    s = s[["model"]+aspects]
    u = u[["model"]+aspects]
    results = pd.DataFrame(columns=aspects)
    delta_s, delta_u = {aspect: [] for aspect in aspects}, {aspect: [] for aspect in aspects}
    for model in s["model"].unique():
        for aspect in aspects:
            baseline = b.loc[b["model"] == model, aspect].item()
            supervised = s.loc[s["model"] == model, aspect].item()
            unsupervised = u.loc[u["model"] == model, aspect].item()
            delta_s[aspect].append(supervised - baseline)
            delta_u[aspect].append(unsupervised - baseline)
    for aspect in aspects:
        delta_s[aspect] = sum(delta_s[aspect]) / len(delta_s[aspect])
        delta_u[aspect] = sum(delta_u[aspect]) / len(delta_u[aspect])
    results.loc["unsupervised"] = delta_u
    results.loc["supervised"] = delta_s
    all_results.append(results)
    print(dataset)
    print(results)