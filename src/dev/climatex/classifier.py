import dill
from dev.constants import data_storage
import torch as t
import pandas as pd
from sklearn.linear_model import LogisticRegression
from random import sample


cache, data = [], []
for report in ["AR3", "AR4", "AR5", "AR6"]:
    x1 = t.load(f"{data_storage}/climatex/activations/{report}_1.pt", pickle_module=dill)
    x2 = t.load(f"{data_storage}/climatex/activations/{report}_2.pt", pickle_module=dill)
    cache.append(x1-x2)
    df = pd.read_json(f"{data_storage}/climatex/prompts/{report}.jsonl", orient="records", lines=True)
    data.append(df)
x1, x2 = x1 - x1.mean(dim=0), x2 - x2.mean(dim=0)
x = t.concat(cache)
data = pd.concat(data).reset_index(drop=True)
y = t.from_numpy(data["true"].to_numpy()) * 2

ar3_ids = list(set([id for id in data["S1"] if "AR3" in id]))
ar4_ids = list(set([id for id in data["S1"] if "AR4" in id]))
ar5_ids = list(set([id for id in data["S1"] if "AR5" in id]))
ar6_ids = list(set([id for id in data["S1"] if "AR6" in id]))
# sample 100 ids for training (25 from each report)
train_ids = []
for ids in [ar3_ids, ar4_ids, ar5_ids, ar6_ids]: train_ids.extend(sample(ids, k=25))
test_ids = []
for ids in [ar3_ids, ar4_ids, ar5_ids, ar6_ids]: test_ids.extend([id for id in ids if id not in train_ids])
# indices of training and test data
train_ixs = data.loc[data["S1"].isin(train_ids)].index.to_numpy()
test_ixs = data.loc[data["S1"].isin(test_ids)].index.to_numpy()
# train/test split
x_train, x_test = x[train_ixs], x[test_ixs]
y_train, y_test = y[train_ixs], y[test_ixs]

lr = LogisticRegression(max_iter=1000, class_weight="balanced")
lr.fit(x_train, y_train)
print(lr.score(x_train, y_train))
print(lr.score(x_test, y_test))
path = "/gws/nopw/j04/ai4er/users/maiush/PPairS/src/dev/climatex"
with open(f"{path}/clf.pkl", "wb") as f: dill.dump(lr, f)
with open(f"{path}/testIDs.pkl", "wb") as f: dill.dump(test_ids, f)