from PPairS.constants import data_path, results_path, collated_results_path
from PPairS.utils import models, dataset_aspects
import sys
from pathlib import Path
import pandas as pd
import torch as t
from sklearn.decomposition import PCA
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
    scores, eigs = [], []
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
        x_train, x_test = x[:split_ix], x[split_ix:]
        y_train, y_test = y[:split_ix], y[split_ix:]

        # fit model
        pca = PCA(1)
        pca.fit(x_train)
        eigs.append(pca.components_[0])
        preds = t.tensor(pca.transform(x_test).squeeze(1))
        p1 = (preds > 0).to(t.int64) + 1
        p2 = (preds < 0).to(t.int64) + 1
        score = max(
            f1(y_test, p1, labels=[1, 2]),
            f1(y_test, p2, labels=[1, 2])
        )
        scores.append(score)
    eigs = t.stack([t.Tensor(x) for x in eigs], dim=0)
    eigs_path = f"{collated_results_path}/{dataset}/{model}"
    Path(eigs_path).mkdir(parents=True, exist_ok=True)
    t.save(eigs, f"{eigs_path}/probe_u.pt")
    results.loc[len(results)] = [model] + scores
results["avg_f1"] = results[aspects].mean(axis=1)
results.sort_values(by=["avg_f1"], ascending=False).to_json(
    f"{collated_results_path}/{dataset}/probe_u_results.jsonl",
    orient="records",
    lines=True
)
print(dataset)
print(results)