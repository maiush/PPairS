from PPairS.constants import data_path, collated_results_path
from PPairS.utils import models, dataset_aspects
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score as kappa


def scores2comparisons(
        true_scores: pd.DataFrame,
        true_comparisons: pd.DataFrame,
        model_scores: pd.DataFrame
) -> pd.DataFrame:
    def compare_scores(x: float, y: float) -> int:
        if x > y: return 1
        elif x < y: return 2
        else: return -1
    compare_func = np.frompyfunc(compare_scores, nin=2, nout=1)

    comparisons, id_columns = [], true_comparisons.columns[:2].tolist()
    for c in id_columns: model_scores[c] = true_scores[c]
    for _, row in true_comparisons.iterrows():
        row = row.tolist()
        context = row[0]
        item1 = row[1][0]
        item2 = row[1][1]
        item1 = model_scores.loc[
            (model_scores[id_columns[0]] == context)&(model_scores[id_columns[1]] == item1),
            model_scores.columns
        ].reset_index(drop=True)
        item2 = model_scores.loc[
            (model_scores[id_columns[0]] == context)&(model_scores[id_columns[1]] == item2),
            model_scores.columns
        ].reset_index(drop=True)
        comparisons.append(item1.combine(item2, compare_func))
    comparisons = pd.concat(comparisons).reset_index(drop=True)
    return comparisons


for dataset in ["newsroom", "summeval", "hanna"]:
    aspects = dataset_aspects[dataset]

    true_scores = pd.read_json(f"{data_path}/{dataset}.jsonl", orient="records", lines=True)
    true_comparisons = pd.read_json(f"{data_path}/{dataset}_pairwise_comparisons.jsonl", orient="records", lines=True)

    results = pd.DataFrame(columns=["model", "method"]+aspects)
    for model in list(models.keys()):
        model_scores = pd.read_json(f"{collated_results_path}/{dataset}/{model}/score.jsonl", orient="records", lines=True)
        model_comparisons = pd.read_json(f"{collated_results_path}/{dataset}/{model}/compare.jsonl", orient="records", lines=True)

        def p2class(p: float) -> int:
            if p > 0.5: return 1
            elif p < 0.5: return 2
            else: return -1
        pairwise_comparisons = model_comparisons.map(p2class)
        direct_scoring = scores2comparisons(true_scores, true_comparisons, model_scores)
        kappa_pc, kappa_ds = [], []
        for aspect in aspects:
            kappa_pc.append(kappa(pd.to_numeric(true_comparisons[aspect]), pd.to_numeric(pairwise_comparisons[aspect]), labels=[1, 2, -1]))
            kappa_ds.append(kappa(pd.to_numeric(true_comparisons[aspect]), pd.to_numeric(direct_scoring[aspect]), labels=[1, 2, -1]))
        results.loc[len(results)] = [model, "direct_scoring"] + kappa_ds
        results.loc[len(results)] = [model, "pairwise_comparisons"] + kappa_pc
    results["avg_kappa"] = results[aspects].mean(axis=1)
    results.sort_values(by=["avg_kappa"], ascending=False).to_json(
        f"{collated_results_path}/{dataset}/baseline_results.jsonl",
        orient="records",
        lines=True
    )