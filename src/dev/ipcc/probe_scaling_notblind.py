from dev.constants import gdrive_path

import os, sys, gc, pickle
import torch as t
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from tqdm.notebook import trange


long_path = f"{gdrive_path}/ipcc/long"
spm_path = f"{gdrive_path}/ipcc/summary"
file, report_type = sys.argv[1], "long"
path = long_path if report_type == "long" else spm_path

# number of unique statements seen during training
n_train = int(sys.argv[2])

# read all prompts used
files = [f for f in os.listdir(f"{path}/prompts") if file in f]
all_prompts = []
for f in files:
    df = pd.read_json(f"{path}/prompts/{f}", orient="records", lines=True)
    df["file"] = f[11]
    df["original_ix"] = range(len(df))
    all_prompts.append(df)
prompts = pd.concat(all_prompts)

scores = []
for _ in range(100):
    # choose a subset of prompts for probe training
    subset = np.random.choice(prompts["S1"].unique(), size=n_train, replace=False)
    prompts = prompts.loc[(prompts["S1"].isin(subset))&(prompts["S2"].isin(subset))].sort_values(by=["file", "original_ix"])

    # load the activations corresponding to this chosen training subset
    choice1, choice2 = [], []
    for file_ix in np.sort(prompts.file.unique()):
        _choice1 = t.load(f"{path}/activations/{file}_PART{file_ix}_CHOICE1.pt")
        _choice2 = t.load(f"{path}/activations/{file}_PART{file_ix}_CHOICE2.pt")
        ixs = prompts.loc[prompts["file"] == file_ix, "original_ix"].values
        choice1.append(_choice1[ixs]); choice2.append(_choice2[ixs])
        del _choice1, _choice2
        gc.collect()
    choice1, choice2 = t.concat(choice1, dim=0), t.concat(choice2, dim=0)

    # shuffle, fit, and score a logistic regression classifier
    perm = t.randperm(len(prompts))
    X = (choice1 - choice2)[perm]
    y = (prompts["P(S1)"] > 0.5).values[perm]
    split = int(0.5*len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    lr = LogisticRegression(max_iter=10_000)
    lr.fit(X_train, y_train)
    score = lr.score(X_test, y_test)

    # cleanup
    del X, y, X_train, y_train, X_test, y_test, lr
    gc.collect()

    scores.append(score)

with open(f"{path}/scaling_notblind/{file}_{n_train}.pkl", "wb") as f:
    pickle.dump(scores, f)