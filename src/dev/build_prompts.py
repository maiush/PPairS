from PPairS.constants import data_path
from PPairS.prompts import descriptions, instructions

import pandas as pd


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

def zero_shot(
        row: pd.Series,
        aspect: str,
        dataset: str,
        context_name: str,
        item_name: str
) -> str:
    template = instructions[dataset][0]
    description = descriptions[dataset][aspect]
    prompt = template.format(
        CONTEXT=row[context_name],
        ITEM=row[item_name],
        DESCRIPTION=description,
        ASPECT=aspect
    )
    return prompt

def compare(
        row: pd.Series,
        aspect: str,
        dataset: str,
        context_name: str,
        item_name: str
) -> str:
    template = instructions[dataset][1]
    description = descriptions[dataset][aspect]
    prompt = template.format(
        CONTEXT=row[context_name],
        ITEM1=row[f"{item_name}1"],
        ITEM2=row[f"{item_name}2"],
        DESCRIPTION=description,
        ASPECT=aspect
    )
    return prompt


for dataset in ["newsroom", "summeval", "hanna"]:
    data = pd.read_json(f"{data_path}/{dataset}.jsonl", orient="records", lines=True)
    for aspect in list(descriptions[dataset].keys()):
        data[aspect] = data.apply(
            lambda row: zero_shot(
                row,
                aspect,
                dataset,
                context_names[dataset],
                item_names[dataset]
            ), axis=1
        )
    data.to_json(f"{data_path}/{dataset}_prompts_zero_shot.jsonl", orient="records", lines=True)


for dataset in ["newsroom", "summeval", "hanna"]:
    data = pd.read_json(f"{data_path}/{dataset}_pairwise_comparisons.jsonl", orient="records", lines=True)
    for aspect in list(descriptions[dataset].keys()):
        data[aspect] = data.apply(
            lambda row: compare(
                row,
                aspect,
                dataset,
                context_names[dataset],
                item_names[dataset]
            ), axis=1
        )
    data.to_json(f"{data_path}/{dataset}_prompts_compare.jsonl", orient="records", lines=True)