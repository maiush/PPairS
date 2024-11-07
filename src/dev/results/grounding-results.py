from PPairS.constants import data_path, collated_results_path
from PPairS.utils import models
import pandas as pd
from sklearn.metrics import f1_score as f1


def p2class(p: float) -> int:
    if p > 0.5: return 1
    elif p < 0.5: return 2
    else: return -1

if __name__ == '__main__':
    for dataset, c_true in zip(['caters', 'mctaco'], ['first', 'correct']):
        results = pd.DataFrame(columns=['model', 'f1'])
        for model in models.keys():
            model_comparisons = pd.read_json(f'{collated_results_path}/{dataset}/{model}/compare.jsonl', orient='records', lines=True)
            true_comparisons = pd.read_json(f"{data_path}/{dataset}_pairwise_comparisons.jsonl", orient="records", lines=True)
            pairwise_comparisons = model_comparisons.map(p2class)
            score = f1(pd.to_numeric(true_comparisons[c_true]), pd.to_numeric(pairwise_comparisons['prob']), labels=[1, 2, -1], average="weighted")
            results.loc[len(results)] = [model, score]
        results.to_json(
            f"{collated_results_path}/{dataset}/results.jsonl",
            orient="records",
            lines=True
        )