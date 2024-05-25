from dev.constants import gdrive_path
from PPairS.prompts import chat_templates, cx_template

from pathlib import Path
import pandas as pd
from datasets import load_dataset

from tqdm import tqdm, trange


# https://huggingface.co/datasets/rlacombe/ClimateX
dataset = load_dataset("rlacombe/ClimateX", split="train").to_pandas()
dataset = dataset.loc[dataset["split"] == "test", ["statement", "score"]].reset_index(drop=True)

# comparisons
prompts = pd.DataFrame(columns=["statementID", "prompt", "P(Statement 1)"])
pairs = []
bar = tqdm(total=len(dataset)**2)
for id1 in range(len(dataset)):
    statement1 = dataset.at[id1, "statement"]
    score1 = dataset.at[id1, "score"]
    for id2 in range(len(dataset)):
        statement2 = dataset.at[id2, "statement"]
        score2 = dataset.at[id2, "score"]
        bar.update()
        if id1 == id2: continue
        if (id1, id2) not in pairs:
            pairs.append((id1, id2))
            pairs.append((id1, id2))
            row = [(id1, id2)]
            prompt = chat_templates["llama3"].format(
                INSTRUCTION=cx_template.format(
                    STATEMENT1=statement1,
                    STATEMENT2=statement2
                ),
                ANSWER="Between Statement 1 and Statement 2, the more confident choice is Statement "
            )
            row.append(prompt)
            if score1 > score2:
                row.append(1)
            elif score1 < score2:
                row.append(0)
            else:
                row.append(0.5)
            prompts.loc[len(prompts)] = row

# contrast pairs
_prompts = prompts.copy()
for choice in [1, 2]:
    for i in trange(len(_prompts), desc=str(choice)):
        prompts.at[i, "prompt"] = f"{_prompts.at[i, 'prompt']}{choice}"
    prompts.to_json(f"{gdrive_path}/climatex/prompts_{choice}.jsonl", orient="records", lines=True)