from PPairS.constants import data_path, collated_results_path
from PPairS.utils import models, dataset_aspects
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score as f1


def scores2comparisons(
        true_scores: pd.DataFrame,
        true_comparisons: pd.DataFrame,
        model_scores: pd.DataFrame
) -> pd.DataFrame:
    '''
    take a direct scoring method, and calculate the implied pairwise comparisons
    1: choice 1
    2: choice 2
    -1: equal / neither
    '''
    def compare_scores(x: float, y: float) -> int:
        if x > y: return 1
        elif x < y: return 2
        else: return -1
    compare_func = np.frompyfunc(compare_scores, nin=2, nout=1)

    comparisons, id_columns = [], true_comparisons.columns[:2].tolist()
    for c in id_columns: model_scores[c] = true_scores[c]
    # iterate through each of the comparisons performed 
    # we want a 1-1 correspondence with the true comparisons
    for _, row in true_comparisons.iterrows():
        row = row.tolist()
        context = row[0]
        item1 = row[1][0]
        item2 = row[1][1]
        # create a separate dataframe of the first item in each comparison...
        item1 = model_scores.loc[
            (model_scores[id_columns[0]] == context)&(model_scores[id_columns[1]] == item1),
            model_scores.columns
        ].reset_index(drop=True)
        # ...and another for the second item in each comparison
        item2 = model_scores.loc[
            (model_scores[id_columns[0]] == context)&(model_scores[id_columns[1]] == item2),
            model_scores.columns
        ].reset_index(drop=True)
        # apply our comparison function to obtain the results
        comparisons.append(item1.combine(item2, compare_func))
    comparisons = pd.concat(comparisons).reset_index(drop=True)
    return comparisons


for dataset in ["newsroom", "summeval", "hanna"]:
    aspects = dataset_aspects[dataset]

    true_scores = pd.read_json(f"{data_path}/{dataset}.jsonl", orient="records", lines=True)
    true_comparisons = pd.read_json(f"{data_path}/{dataset}_pairwise_comparisons.jsonl", orient="records", lines=True)

    # we will store a single dataframe with all baseline results, to make things easy
    # this includes direct scoring, g-eval, and pairwise comparisons (via zero-shot prompting)
    results = pd.DataFrame(columns=["model", "method"]+aspects)
    for model in list(models.keys()):
        model_scores = pd.read_json(f"{collated_results_path}/{dataset}/{model}/score.jsonl", orient="records", lines=True)
        model_geval = pd.read_json(f"{collated_results_path}/{dataset}/{model}/geval.jsonl", orient="records", lines=True)
        model_comparisons = pd.read_json(f"{collated_results_path}/{dataset}/{model}/compare.jsonl", orient="records", lines=True)
        # we're just storing the actual comparisons, so we need to convert the probabilities (see results.py) to these comparison values
        def p2class(p: float) -> int:
            if p > 0.5: return 1
            elif p < 0.5: return 2
            else: return -1
        pairwise_comparisons = model_comparisons.map(p2class)
        direct_scoring = scores2comparisons(true_scores, true_comparisons, model_scores)
        g_eval = scores2comparisons(true_scores, true_comparisons, model_geval)
        f1_pc, f1_ds, f1_g = [], [], []
        for aspect in aspects:
            # calculate f1 score for all baseline methods
            f1_ds.append(f1(pd.to_numeric(true_comparisons[aspect]), pd.to_numeric(direct_scoring[aspect]), labels=[1, 2, -1], average="weighted"))
            f1_g.append(f1(pd.to_numeric(true_comparisons[aspect]), pd.to_numeric(g_eval[aspect]), labels=[1, 2, -1], average="weighted"))
            f1_pc.append(f1(pd.to_numeric(true_comparisons[aspect]), pd.to_numeric(pairwise_comparisons[aspect]), labels=[1, 2, -1], average="weighted"))
        results.loc[len(results)] = [model, "direct_scoring"] + f1_ds
        results.loc[len(results)] = [model, "g_eval"] + f1_g
        results.loc[len(results)] = [model, "pairwise_comparisons"] + f1_pc
    results["avg_f1"] = results[aspects].mean(axis=1)
    results.sort_values(by=["avg_f1"], ascending=False).to_json(
        f"{collated_results_path}/{dataset}/baseline_results.jsonl",
        orient="records",
        lines=True
    )