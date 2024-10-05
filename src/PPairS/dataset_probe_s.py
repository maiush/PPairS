from PPairS.constants import data_path, results_path, collated_results_path
from PPairS.utils import models, dataset_aspects
import pandas as pd
import torch as t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score as f1
from tqdm import tqdm

models = list(models.keys())
# train/test split
split = 0.5


datasets = ["newsroom", "summeval", "hanna"]
results = pd.DataFrame(columns=["model"]+datasets)
for model in tqdm(models):
    feats, scores = [], []
    for dataset in datasets:
        aspects = dataset_aspects[dataset]
        data = pd.read_json(
            f"{data_path}/{dataset}_pairwise_comparisons.jsonl",
            orient="records",
            lines=True
        )
        X, Y = [], []
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
            X.append(x); Y.append(y)
        X = t.cat(tuple(X), dim=0)
        Y = t.cat(tuple(Y), dim=0)
        # random shuffle 
        perm = t.randperm(len(X))
        X, Y = X[perm], Y[perm]
        # train/test split
        split_ix = int(split*len(X))
        x_train, x_test = X[:split_ix], X[split_ix:]
        y_train, y_test = Y[:split_ix], Y[split_ix:]
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
    results.loc[len(results)] = [model] + scores
    feats = t.stack([t.Tensor(x) for x in feats], dim=0)
    t.save(feats, f"{collated_results_path}/{model}_probe_s.pt")
results.to_json(
    f"{collated_results_path}/dataset_probe_s.jsonl",
    orient="records",
    lines=True
)
print(dataset)
print(results)