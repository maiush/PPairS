from dev.constants import gdrive_path

import os, sys, gc, pickle
import torch as t
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from tqdm import trange


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
for _ in trange(100):

    # choose a subset of prompts for probe training
    subset = np.random.choice(prompts["S1"].unique(), size=n_train, replace=False)
    train_prompts = prompts.loc[(prompts["S1"].isin(subset))&(prompts["S2"].isin(subset))].sort_values(by=["file", "original_ix"])

    # load the activations corresponding to this chosen training subset
    choice1, choice2 = [], []
    for file_ix in np.sort(train_prompts.file.unique()):
        _choice1 = t.load(f"{path}/activations/{file}_PART{file_ix}_CHOICE1.pt")
        _choice2 = t.load(f"{path}/activations/{file}_PART{file_ix}_CHOICE2.pt")
        ixs = train_prompts.loc[train_prompts["file"] == file_ix, "original_ix"].values
        choice1.append(_choice1[ixs]); choice2.append(_choice2[ixs])
        del _choice1, _choice2
        gc.collect()
    choice1, choice2 = t.concat(choice1, dim=0), t.concat(choice2, dim=0)

    # shuffle and fit a logistic regression classifier
    perm = t.randperm(len(train_prompts))
    X_train = (choice1 - choice2)[perm]
    y_train = (train_prompts["P(S1)"] > 0.5).values[perm]
    lr = LogisticRegression(max_iter=10_000)
    lr.fit(X_train, y_train)

    # cleanup
    del train_prompts, choice1, choice2, X_train, y_train
    gc.collect()

    # choose a different subset of prompts for probe testing
    subset = np.random.choice([s for s in prompts["S1"].unique() if s not in subset], size=100, replace=False)
    test_prompts = prompts.loc[(prompts["S1"].isin(subset))&(prompts["S2"].isin(subset))].sort_values(by=["file", "original_ix"])

    # load the activations corresponding to this chosen test subset
    choice1, choice2 = [], []
    for file_ix in np.sort(test_prompts.file.unique()):
        _choice1 = t.load(f"{path}/activations/{file}_PART{file_ix}_CHOICE1.pt")
        _choice2 = t.load(f"{path}/activations/{file}_PART{file_ix}_CHOICE2.pt")
        ixs = test_prompts.loc[test_prompts["file"] == file_ix, "original_ix"].values
        choice1.append(_choice1[ixs]); choice2.append(_choice2[ixs])
        del _choice1, _choice2
        gc.collect()
    choice1, choice2 = t.concat(choice1, dim=0), t.concat(choice2, dim=0)

    # score the classifier
    X_test = choice1 - choice2
    y_test = (test_prompts["P(S1)"] > 0.5).values
    score = lr.score(X_test, y_test)

    # cleanup
    del test_prompts, choice1, choice2, X_test, y_test
    gc.collect()

    scores.append(score)

with open(f"{path}/scaling/{file}_{n_train}.pkl", "wb") as f:
    pickle.dump(scores, f)