import sys
from dev.constants import gdrive_path
from PPairS.sort import mergesort
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr as rho

import torch as t

from copy import deepcopy
from typing import List, Tuple, Optional
from tqdm import tqdm, trange


def train_probe(model: str, dataset: str, aspect: str, split: float=0.5) -> Tuple[np.ndarray]:
    # load activations
    act_path = f"{gdrive_path}/benchmarks/activations/{model}/{dataset}_{aspect}"
    x_pos = t.load(f"{act_path}_1.pt")
    x_neg = t.load(f"{act_path}_2.pt")
    x = t.concat([x_pos - x_neg, x_neg - x_pos])

    # load labels and mapping to IDs
    probe_prompts_path = f"{gdrive_path}/benchmarks/prompts_short/{model}/{dataset}_mine_1.jsonl"
    prompts = pd.read_json(probe_prompts_path, orient="records", lines=True)
    id2ix = prompts.apply(lambda row: [row["article_id"], row["summary_id"][0], row["summary_id"][1]], axis=1).to_list()
    id2ix = np.concatenate([np.array(id2ix), np.array([(a, s2, s1) for (a, s1, s2) in id2ix])])
    def get_label(scores: List[float]) -> float:
        label = 0.5
        if scores[0] > scores[1]: label = 1.
        if scores[0] < scores[1]: label = 0.
        return label
    y = prompts[aspect].apply(get_label).to_numpy()
    y = np.concatenate([y, 1-y])

    # train/test split
    perm = t.randperm(len(x))
    split_ix = int(split*len(x))
    x_train, x_test = x[perm[:split_ix]], x[perm[split_ix:]]
    y_train, y_test = y[perm[:split_ix]], y[perm[split_ix:]]
    id2ix_train, id2ix_test = id2ix[perm[:split_ix]], id2ix[perm[split_ix:]]

    # fit model (converting y as probabilities into class labels)
    lr = LogisticRegression(max_iter=10_000, n_jobs=-1)
    lr.fit(x_train, y_train*2)

    # obtain test predictions (converting back to probabilities)
    predictions = lr.predict(x_test) / 2

    # return predictions and corresponding labels
    return predictions, id2ix_test


def collect_results(model: str, dataset: str, aspect: str) -> pd.DataFrame: 
    results = pd.DataFrame(columns=["articleID", "S1", "S2", "true", "direct-scoring", "logits", "probe"])

    # load scores
    scores_path = f"{gdrive_path}/benchmarks/scores/{model}/{dataset}_{aspect}.jsonl"
    scores = pd.read_json(scores_path, orient="records", lines=True)
    # load pairs method
    logits_path = f"{gdrive_path}/benchmarks/_logits/{model}/{dataset}_{aspect}.jsonl"
    logits = pd.read_json(logits_path, orient="records", lines=True)
    logits["S1"] = logits["summary_id"].apply(lambda x: x[0])
    logits["S2"] = logits["summary_id"].apply(lambda x: x[1])
    logits = logits[["article_id", "S1", "S2", "p_s1", "p_s2"]]
    _logits = logits.copy()
    _logits = _logits.rename({"S1": "S2", "S2": "S1", "p_s1": "p_s2", "p_s2": "p_s1"}, axis=1)
    logits = pd.concat([logits, _logits])
    # load ppairs method
    probe_predictions, probe_ixs = train_probe(model, dataset, aspect, 0.5)

    # collect results
    for article in scores["article_id"].unique():
        subset = scores.loc[scores["article_id"] == article]
        ids = subset["summary_id"].unique()
        for id1 in ids:
            for id2 in ids:
                row1, row2 = subset.loc[subset["summary_id"] == id1], subset.loc[subset["summary_id"] == id2]
                
                # ground-truth
                gt = 0.5
                s1, s2 = row1[aspect].item(), row2[aspect].item()
                if s1 > s2: gt = 1
                elif s1 < s2: gt = 0

                # direct-scoring
                ds = 0.5
                s1, s2 = row1["score"].item(), row2["score"].item()
                if s1 > s2: ds = 1
                elif s1 < s2: ds = 0

                # logits
                condition = (logits["article_id"] == article) & (logits["S1"] == id1) & (logits["S2"] == id2)
                if len(logits.loc[condition]) == 0: lgt = np.nan
                else: lgt = logits.loc[condition, "p_s1"].item()

                # probe
                ix = np.where(np.all(probe_ixs == [article, id1, id2], axis=1))[0]
                if len(ix) == 0: prb = np.nan
                else: prb = probe_predictions[ix].item()

                results.loc[len(results)] = [article, id1, id2, gt, ds, lgt, prb]
                
    return results.dropna()


def get_U(P):
    U = np.zeros_like(P)
    for i in range(len(U)):
        for j in range(len(U)):
            U[i, j] = -P[i, j]*np.log(P[i, j] + 1e-10) - P[j, i]*np.log(P[j, i] + 1e-10)
    return U


def get_spearman(model: str, dataset: str, aspect: str, results: pd.DataFrame) -> dict:
    # load scores
    scores_path = f"{gdrive_path}/benchmarks/scores/{model}/{dataset}_{aspect}.jsonl"
    scores = pd.read_json(scores_path, orient="records", lines=True)

    correlations_ds, correlations_lgt, correlations_prb = [], [], []
    for article in tqdm(results["articleID"].unique()):
        subset = results.loc[results["articleID"] == article]
        # skeleton for comparison matrices
        ids = list(set(subset["S1"].tolist() + subset["S2"].tolist()))
        P = pd.DataFrame(columns=ids)
        for id in ids: P.loc[id] = [0.5]*len(ids)
        # one comparison matrix for each method
        P_true, P_ds, P_lgt, P_prb = deepcopy(P), deepcopy(P), deepcopy(P), deepcopy(P)
        for _, row in subset.iterrows():
            s1, s2 = row["S1"], row["S2"]
            # ground-truth
            true = row["true"]
            P_true.loc[s1, s2] = true
            P_true.loc[s2, s1] = 1-true
            # direct-scoring
            ds = row["direct-scoring"]
            P_ds.loc[s1, s2] = ds
            P_ds.loc[s2, s1] = 1-ds
            # logits
            lgt = row["logits"]
            P_lgt.loc[s1, s2] = lgt
            P_lgt.loc[s2, s1] = 1-lgt
            # probe
            prb = row["probe"]
            P_prb.loc[s1, s2] = prb
            P_prb.loc[s2, s1] = 1-prb
        P_true, P_ds, P_lgt, P_prb = P_true.to_numpy(), P_ds.to_numpy(), P_lgt.to_numpy(), P_prb.to_numpy()

        true_ranking = mergesort(
            ixs = np.arange(len(ids)),
            P = P_true,
            beam = True,
            beam_size = 1000,
            Uh = 0.6,
            U = get_U(P_true)
        )

        ds_ranking = mergesort(
            ixs = np.arange(len(ids)),
            P = P_ds,
            beam = True,
            beam_size = 1000,
            Uh = 0.6,
            U = get_U(P_ds)
        )

        lgt_ranking = mergesort(
            ixs = np.arange(len(ids)),
            P = P_lgt,
            beam = True,
            beam_size = 1000,
            Uh = 0.6,
            U = get_U(P_lgt)
        )

        prb_ranking = mergesort(
            ixs = np.arange(len(ids)),
            P = P_prb,
            beam = True,
            beam_size = 1000,
            Uh = 0.6,
            U = get_U(P_prb)
        )

        score_values = scores.loc[scores["article_id"] == article].set_index("summary_id").loc[ids, aspect].to_numpy()
        r_ds = rho(score_values[ds_ranking], np.sort(score_values)[::-1])[0]
        r_lgt = rho(score_values[lgt_ranking], np.sort(score_values)[::-1])[0]
        r_prb = rho(score_values[prb_ranking], np.sort(score_values)[::-1])[0]


        correlations_ds.append(r_ds)
        correlations_lgt.append(r_lgt)
        correlations_prb.append(r_prb)

    spearman = {}
    for m, c in zip(["direct-scoring", "logits", "probe"], [correlations_ds, correlations_lgt, correlations_prb]):
        spearman[m] = sum(c) / len(c)

    return spearman


def get_all_results(model: str, dataset: str, aspect: str) -> Tuple[dict]:
    # collect results
    results = collect_results(model, dataset, aspect)

    # calculate accuracies
    accuracy = {}
    for c in ["direct-scoring", "logits", "probe"]:
        accuracy[c] = ((results[c] > 0.5) == (results["true"] > 0.5)).mean().item()

    # calculate spearman correlation
    spearman = get_spearman(model, dataset, aspect, results)

    return (accuracy, spearman)

aspects = ["coherence", "consistency", "fluency", "relevance"]
model, dataset = sys.argv[1:3]
accuracy = pd.DataFrame(columns=["direct-scoring", "logits", "probe"])
spearman = pd.DataFrame(columns=["direct-scoring", "logits", "probe"])
for aspect in aspects:
    a, s = get_all_results(model, dataset, aspect)
    accuracy.loc[aspect] = [a["direct-scoring"], a["logits"], a["probe"]]
    spearman.loc[aspect] = [s["direct-scoring"], s["logits"], s["probe"]]
print(accuracy)
print(spearman)