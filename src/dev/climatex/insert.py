import os, dill
from dev.constants import gdrive_path

import torch as t
import pandas as pd
from scipy.stats import mode

from random import sample
from typing import Optional
from tqdm import tqdm


cache, data = [], []
for report in ["AR3", "AR4", "AR5", "AR6"]:
    x1 = t.load(f"{gdrive_path}/climatex/activations/{report}_1.pt", pickle_module=dill)
    x2 = t.load(f"{gdrive_path}/climatex/activations/{report}_2.pt", pickle_module=dill)
    cache.append(x1-x2)
    df = pd.read_json(f"{gdrive_path}/climatex/prompts/{report}.jsonl", orient="records", lines=True)
    data.append(df)
x1, x2 = x1 - x1.mean(dim=0), x2 - x2.mean(dim=0)
x = t.concat(cache)
data = pd.concat(data).reset_index(drop=True)
y = t.from_numpy(data["true"].to_numpy()) * 2

path = "/gws/nopw/j04/ai4er/users/maiush/PPairS/src/dev/climatex"
with open(f"{path}/clf.pkl", "rb") as f: clf = dill.load(f)
with open(f"{path}/testIDs.pkl", "rb") as f: test_ids = dill.load(f)

files = os.listdir(f"{gdrive_path}/climatex/claims")
tagged_claims = []
for file in files:
    df = pd.read_json(f"{gdrive_path}/climatex/claims/{file}", orient="records", lines=True)
    if len(df) == 0: continue
    tagged_claims.append(df)
tagged_claims = pd.concat(tagged_claims).reset_index(drop=True)

tagged_test = tagged_claims.loc[tagged_claims["statementID"].isin(test_ids)]
tagged_ar6 = tagged_claims.loc[tagged_claims["statementID"].isin(data["S2"])]

test_cache = {tag: tagged_test.loc[tagged_test["tag"] == tag, "statementID"].unique().tolist() for tag in tagged_test["tag"].unique()}
ar6_cache = {tag: tagged_ar6.loc[tagged_ar6["tag"] == tag, "statementID"].unique().tolist() for tag in tagged_ar6["tag"].unique()}

def get_tag(id: str, cache: dict) -> str:
    for tag, claims in cache.items():
        if id in claims: return tag

def predict(id: str, test: str, N_sample: int=10) -> float:
    subset = ar6_cache[test]
    subset = sample(subset, min(N_sample, len(subset)))
    condition = (data["S1"] == id) & (data["S2"].isin(subset))
    ixs = data.loc[condition].index.tolist()
    acts = x[ixs]
    return mode(clf.predict(acts))[0] / 2.

tags = ["low", "medium", "high", "very_high"]
def ppairs_insert(id: str, low: Optional[int]=None, high: Optional[int]=None, N_sample: Optional[int]=10) -> str:
    if low is None: low = 0
    if high is None: high = len(tags)-1

    mid = (low + high) // 2
    tag = tags[mid]
    comparison = predict(id, tag, N_sample)
    if comparison == 1.:
        if mid == high: return tag
        else: return ppairs_insert(id, mid+1, high, N_sample)
    elif comparison == 0.:
        if mid == low: return tag
        else: return ppairs_insert(id, low, mid-1, N_sample)
    elif comparison == 0.5:
        return tag
    
accuracy, bar = [], tqdm(test_ids)
for id in bar:
    true = get_tag(id, test_cache)
    prediction = ppairs_insert(id)
    accuracy.append(prediction == true)
    bar.set_description(f"{round(sum(accuracy)/len(accuracy), 3)}")

for report in ["AR3", "AR4", "AR5", "AR6"]:
    test_set = [id for id in test_ids if report in id]
    accuracy, bar = [], tqdm(test_set)
    for id in bar:
        true = get_tag(id, test_cache)
        prediction = ppairs_insert(id)
        accuracy.append(prediction == true)
        bar.set_description(f"{report}: {round(sum(accuracy)/len(accuracy), 3)}")