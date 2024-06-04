from dev.constants import gdrive_path
from PPairS.prompts import chat_templates, ipcc_template

import sys, gc
import pandas as pd
from tqdm import tqdm


long_path = f"{gdrive_path}/ipcc/long"
spm_path = f"{gdrive_path}/ipcc/summary"

file, type = sys.argv[1:3]
inpath = f"{long_path}/long_parsed" if type == "long" else f"{spm_path}/summary_parsed"
outpath = f"{long_path}/prompts" if type == "long" else f"{spm_path}/prompts"
claims = pd.read_json(f"{inpath}/{file}_processed.jsonl", orient="records", lines=True)

tag_scores = ["very_low", "low", "medium", "high", "very_high"]
prompts = pd.DataFrame(columns=["C1", "C2", "S1", "S2", "prompt", "P(S1)"])
pairs = []
file_counter = 1
bar = tqdm(total=len(claims)**2)
for i in range(len(claims)):
    for j in range(len(claims)):
        c1_id, c1 = claims.iloc[i, [0, 1]].values
        c2_id, c2 = claims.iloc[j, [0, 1]].values
        s1_id, s1 = claims.iloc[i, [2, 3]].values
        s2_id, s2 = claims.iloc[j, [2, 3]].values
        bar.update()
        if s1 == s2: continue
        if (s1, s2) not in pairs:
            pairs.append((s1, s2))
            pairs.append((s2, s1))

            row = [c1_id, c2_id, s1_id, s2_id]
            prompt = chat_templates["llama3"].format(
                INSTRUCTION=ipcc_template.format(
                    CONTEXT1=c1,
                    CONTEXT2=c2,
                    STATEMENT1=s1,
                    STATEMENT2=s2
                ),
                ANSWER="Between Statement 1 and Statement 2, the more confident choice is Statement "
            )
            row.append(prompt)
            score1, score2 = tag_scores.index(claims.iloc[i, 4]), tag_scores.index(claims.iloc[j, 4])
            if score1 > score2: row.append(1)
            elif score1 < score2: row.append(0)
            else: row.append(0.5)
            prompts.loc[len(prompts)] = row
            if len(prompts) == 10000:
                outpath = f"{outpath}/{file}_prompts{file_counter}.jsonl"
                prompts.to_json(outpath, orient="records", lines=True)
                del prompts; gc.collect()
                prompts = pd.DataFrame(columns=["C1", "C2", "S1", "S2", "prompt", "P(S1)"])
                file_counter += 1
outpath = f"{outpath}/{file}_prompts{file_counter}.jsonl"
prompts.to_json(outpath, orient="records", lines=True)