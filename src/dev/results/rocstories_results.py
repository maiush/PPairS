from PPairS.constants import data_path, collated_results_path
from PPairS.utils import models, dataset_aspects
import pandas as pd
from sklearn.metrics import f1_score as f1


def p2class(p: float) -> int:
    if p > 0.5: return 1
    elif p < 0.5: return 2
    else: return -1

if __name__ == '__main__':
    results = pd.DataFrame(columns=['model', 'f1'])
    for model in models.keys():
        model_comparisons = pd.read_json(f'{collated_results_path}/rocstories/{model}/compare.jsonl', orient='records', lines=True)
        true_comparisons = pd.read_json(f"{data_path}/rocstories.jsonl", orient="records", lines=True)
        pairwise_comparisons = model_comparisons.map(p2class)
        score = f1(pd.to_numeric(true_comparisons['correct']), pd.to_numeric(pairwise_comparisons['consistency']), labels=[1, 2, -1], average="weighted")
        results.loc[len(results)] = [model, score]
    results.to_json(
        f"{collated_results_path}/rocstories/results.jsonl",
        orient="records",
        lines=True
    )