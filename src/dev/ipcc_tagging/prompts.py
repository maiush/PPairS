import sys
from dev.constants import gdrive_path
from PPairS.prompts import chat_templates, ipcc_template

import numpy as np
import pandas as pd

from typing import Tuple
from tqdm import tqdm


path = f"{gdrive_path}/ipcc_tagging/reports"

ar1 = pd.read_json(f"{path}/AR1_processed.jsonl", orient="records", lines=True)
ar2 = pd.read_json(f"{path}/AR2_processed.jsonl", orient="records", lines=True)
ar3 = pd.read_json(f"{path}/AR3_processed.jsonl", orient="records", lines=True)
ar4 = pd.read_json(f"{path}/AR4_processed.jsonl", orient="records", lines=True)
ar5 = pd.read_json(f"{path}/AR5_processed.jsonl", orient="records", lines=True)
ar6 = pd.read_json(f"{path}/AR6_processed.jsonl", orient="records", lines=True)
reports = [ar1, ar2, ar3, ar4, ar5, ar6]

def get_claim(id: str, tag: bool=False) -> Tuple[str]:
    report_ix = int(id[2])-1
    claims = reports[report_ix]
    row = claims.loc[claims["statementID"] == id]
    assert len(row) == 1
    if tag: return (row["context"].item(), row["statement"].item(), row["tag"].item())
    else: return (row["context"].item(), row["statement"].item())

tags = ["very_low", "low", "medium", "high", "very_high"]


i = int(sys.argv[1])
report = f"AR{i}"
report_ix = int(report[-1])-1
claims = reports[report_ix]

ids = claims["statementID"].unique()
ar6_ids = ar6["statementID"].unique()

prompts, part = pd.DataFrame(columns=["S1", "S2", "P(S1)", "prompt"]), 1
for id1 in tqdm(ids, desc=report):
    for id2 in ar6_ids:
        if i > 2:
            c1, s1, tag1 = get_claim(id1, True)
            c2, s2, tag2 = get_claim(id2, True)
            tag1 = tags.index(tag1)
            tag2 = tags.index(tag2)
            p = 0.5
            if tag1 > tag2: p = 1.
            if tag1 < tag2: p = 0.
        else:
            c1, s1 = get_claim(id1, False)
            c2, s2 = get_claim(id2, False)
            p = np.nan

        prompt = chat_templates["llama3"].format(
            INSTRUCTION=ipcc_template.format(
                CONTEXT1=c1,
                CONTEXT2=c2,
                STATEMENT1=s1,
                STATEMENT2=s2
            ),
            ANSWER="Between Statement 1 and Statement 2, the more confident choice is Statement "
        )
        row = [id1, id2, p, prompt]
        prompts.loc[len(prompts)] = row

    if len(prompts) > 10_000: 
        outpath = f"{gdrive_path}/ipcc_tagging/prompts/{report}_PART{part}.jsonl"
        prompts.to_json(outpath, orient="records", lines=True)
        prompts = pd.DataFrame(columns=["S1", "S2", "P(S1)", "prompt"])
        part += 1
if len(prompts) > 0:
    outpath = f"{gdrive_path}/ipcc_tagging/prompts/{report}_PART{part}.jsonl"
    prompts.to_json(outpath, orient="records", lines=True)