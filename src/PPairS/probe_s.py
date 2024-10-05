from PPairS.constants import data_path, results_path, collated_results_path
from PPairS.utils import models, dataset_aspects
import sys
from pathlib import Path
import pandas as pd
import torch as t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score as f1
from tqdm import tqdm

models = list(models.keys())
# train/test split
split = 0.5


dataset  = sys.argv[1]
aspects = dataset_aspects[dataset]
data = pd.read_json(
    f"{data_path}/{dataset}_pairwise_comparisons.jsonl",
    orient="records",
    lines=True
)

results = pd.DataFrame(columns=["model"]+aspects)
for model in tqdm(models, desc=dataset):
    scores,feats = [], []
    for aspect in aspects:
        # load contrast pair activations
        act_path = f"{results_path}/{dataset}/{model}/{aspect}_contrast"
        x1 = t.load(f"{act_path}_1.pt", weights_only=True).float()
        x2 = t.load(f"{act_path}_2.pt", weights_only=True).float()
        # centering
        x1 -= x1.mean(0)
        x2 -= x2.mean(0)
        # contrast pair differences
        x = x1 - x2
        # labels
        y = t.tensor(data[aspect], dtype=int)
        # mask out equal pairs
        mask = y != -1
        x, y = x[mask], y[mask]
        # random shuffle
        perm = t.randperm(len(x))
        x, y = x[perm], y[perm]
        # train/test split
        split_ix = int(split*len(x))
        # split_ix = 1000
        x_train, x_test = x[:split_ix], x[split_ix:]
        y_train, y_test = y[:split_ix], y[split_ix:]

        # fit model
        lr = LogisticRegression(
            solver="lbfgs",
            fit_intercept=False,
            penalty="l2",
            class_weight="balanced",
            max_iter=10_000,
            n_jobs=-1,
            random_state=123456
        )
        lr.fit(x_train, y_train)
        preds = lr.predict(x_test)
        feats.append(lr.coef_[0])
        score = f1(y_test, preds, labels=[1, 2])
        scores.append(score)
    feats = t.stack([t.Tensor(x) for x in feats], dim=0)
    feats_path = f"{collated_results_path}/{dataset}/{model}"
    Path(feats_path).mkdir(parents=True, exist_ok=True)
    t.save(feats, f"{feats_path}/probe_s.pt")
    results.loc[len(results)] = [model] + scores
results["avg_f1"] = results[aspects].mean(axis=1)
results.sort_values(by=["avg_f1"], ascending=False).to_json(
    f"{collated_results_path}/{dataset}/probe_s_results.jsonl",
    orient="records",
    lines=True
)
print(dataset)
print(results)