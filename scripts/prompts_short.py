'''
Having generated prompt datasets in workbook 001, we want to create smaller datasets for testing.

Since we are looking at pairwise comparisons we will make two assumptions for shortening:
- we don't need to take pairwise comparisons between a given summary and itself.
- having compared summary i to summary j, we don't need to compare summary j to summary i.

Note the latter summary is likely incorrect, or at least needs to be verified at some point.
'''

import os, sys; sys.path.append("../utils")
from constants import *
import pandas as pd
from tqdm import tqdm


for model in ["mistral", "llama2", "llama3"]:
    files = [f for f in os.listdir(f"{gdrive_path}/prompts/{model}") if "score" not in f]
    for file in files:
        df = pd.read_json(f"{gdrive_path}/prompts/{model}/{file}", orient="records", lines=True)
        ixs = []
        for article in tqdm(df["article_id"].unique(), desc=f"{model}:{file}"):
            subset = df.loc[df["article_id"]==article, :]
            pairs = []
            for i, row in subset.iterrows():
                id1, id2 = subset.at[i, "summary_id"]
                if id1 == id2: continue
                if (id2, id1) in pairs: continue
                ixs.append(i)
                pairs.append((id1, id2))
        df = df.loc[ixs]
        df.to_json(f"{gdrive_path}/prompts_short/{model}/{file}", orient="records", lines=True)