from dev.constants import data_path
from PPairS.prompts import descriptions

import pandas as pd
from itertools import combinations


context_names = {
    "newsroom": "article",
    "summeval": "article",
    "hanna": "prompt"
}
item_names = {
    "newsroom": "summary",
    "summeval": "summary",
    "hanna": "story"
}

for dataset in ["newsroom", "summeval", "hanna"]:
    context_name, item_name = context_names[dataset], item_names[dataset]
    data = pd.read_json(f"{data_path}/{dataset}.jsonl", orient="records", lines=True)
    aspects = list(descriptions[dataset].keys())
    pairs = list(combinations(data[f"{item_name}_id"].unique(), 2))
    columns = [
        f"{context_name}_id",
        f"{item_name}_id",
        f"{context_name}",
        f"{item_name}1",
        f"{item_name}2"
    ] + aspects
    prompts = pd.DataFrame(columns=columns)
    for context_id in data[f"{context_name}_id"].unique():
        data_context = data.loc[data[f"{context_name}_id"] == context_id]
        context = data_context[context_name].unique().item()
        for pair in pairs:
            data_item1 = data_context.loc[data[f"{item_name}_id"] == pair[0]]
            data_item2 = data_context.loc[data[f"{item_name}_id"] == pair[1]]
            item1, item2 = data_item1[item_name].item(), data_item2[item_name].item()
            row = [context_id, pair, context, item1, item2]
            for aspect in aspects:
                s1 = data_item1[aspect].item()
                s2 = data_item2[aspect].item()
                if s1 > s2: row.append(1)
                elif s1 < s2: row.append(2)
                elif s1 == s2: row.append(-1)
            prompts.loc[len(prompts)] = row
    print(f"{dataset}: {len(prompts)} comparisons.")
    prompts.to_json(f"{data_path}/{dataset}_pairwise_comparisons.jsonl", orient="records", lines=True)