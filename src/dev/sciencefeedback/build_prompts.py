from dev.constants import gdrive_path
from PPairS.prompts import chat_templates, fc_template_score, fc_template_compare

from pathlib import Path
import numpy as np
import pandas as pd

from tqdm import tqdm, trange


for topic in ["Climate", "Health", "Energy"]:
    print(f"\n{topic}")
    
    v2int = {
        "Incorrect": -1,
        "Misleading": 0,
        "Correct": 1
    }

    path = f"{gdrive_path}/sciencefeedback/sciencefeedback.jsonl"
    df = pd.read_json(path, orient="records", lines=True)
    # we sample a maximum of 100 claims to limit dataset size
    cv = df.loc[df["topic"] == topic, ["topic", "claim", "verdict"]].values[:min(100, len(df))]
    topics, claims, verdicts = cv[:, 0], cv[:, 1], cv[:, 2]
    ids = np.arange(len(claims))

    # scoring
    prompts = pd.DataFrame(columns=["claimID", "topic", "claim", "verdict", "prompt"])
    bar = tqdm(total=len(ids), desc="score")
    for id, topic, claim, verdict in zip(ids, topics, claims, verdicts):
        bar.update()
        prompt = chat_templates["llama3"].format(
            INSTRUCTION=fc_template_score.format(CLAIM=claim),
            ANSWER="The above claim is "
        )
        prompts.loc[len(prompts)] = [id, topic, claim, verdict, prompt]
    outpath = f"{gdrive_path}/sciencefeedback/prompts"
    Path(outpath).mkdir(parents=True, exist_ok=True)
    prompts.to_json(f"{outpath}/{topic.lower()}feedback_score.jsonl", orient="records", lines=True)

    # comparison
    prompts = pd.DataFrame(columns=["claimID", "prompt", "P(Claim 1)"])
    pairs = []
    bar = tqdm(total=len(ids)**2, desc="compare")
    for id1, claim1, v1 in zip(ids, claims, verdicts):
        for id2, claim2, v2 in zip(ids, claims, verdicts):
            bar.update()
            if id1 == id2: continue
            if (id1, id2) not in pairs:
                pairs.append((id1, id2))
                pairs.append((id1, id2))
                row = [(id1, id2)]
                prompt = chat_templates["llama3"].format(
                    INSTRUCTION=fc_template_compare.format(
                        CLAIM1=claim1,
                        CLAIM2=claim2
                    ),
                    ANSWER="Between Claim 1 and Claim 2, the more factually accurate / less ambiguous choice is Claim "
                )
                row.append(prompt)
                if v2int[v1] > v2int[v2]:
                    row.append(1)
                elif v2int[v1] < v2int[v2]:
                    row.append(0)
                else:
                    row.append(0.5)
                prompts.loc[len(prompts)] = row
    prompts.to_json(f"{outpath}/{topic.lower()}feedback_compare.jsonl", orient="records", lines=True)

    # contrast pairs
    _prompts = prompts.copy()
    for choice in [1, 2]:
        for i in trange(len(_prompts), desc=str(choice)):
            prompts.at[i, "prompt"] = f"{_prompts.at[i, 'prompt']}{choice}"
        prompts.to_json(f"{outpath}/{topic.lower()}feedback_contrast_{choice}.jsonl", orient="records", lines=True)