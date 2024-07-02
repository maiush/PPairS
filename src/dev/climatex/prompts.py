import os
from dev.constants import data_storage
from PPairS.prompts import chat_templates, ipcc_template
import pandas as pd


# choose 250 random claims from AR6 to compare them against
files = [f for f in os.listdir(f"{data_storage}/climatex/claims") if "AR6" in f]
ar6_data = []
for file in files: ar6_data.append(pd.read_json(f"{data_storage}/climatex/claims/{file}", orient="records", lines=True))
ar6_data = pd.concat(ar6_data)
ar6_data = ar6_data.sample(n=min(250, len(ar6_data)))

# for each of AR3, AR4, AR5, choose 100 random claims
for i in range(3, 7):
    report = f"AR{i}"
    files = [f for f in os.listdir(f"{data_storage}/climatex/claims") if report in f]
    data = []
    for file in files: data.append(pd.read_json(f"{data_storage}/climatex/claims/{file}", orient="records", lines=True))
    data = pd.concat(data)
    if i == 6:
        ar6_ids = ar6_data["statementID"].unique()
        data = data.loc[~data["statementID"].isin(ar6_ids)]
    data = data.sample(n=min(100, len(data)))

    # build prompts
    tags = ["very_low", "low", "medium", "high", "very_high"]
    prompts = pd.DataFrame(columns=["S1", "S2", "prompt", "true"])
    for id1 in data["statementID"].unique():
        for id2 in ar6_data["statementID"].unique():
            c1 = data.loc[data["statementID"] == id1, "context"].item()
            c2 = ar6_data.loc[ar6_data["statementID"] == id2, "context"].item()
            s1 = data.loc[data["statementID"] == id1, "statement"].item()
            s2 = ar6_data.loc[ar6_data["statementID"] == id2, "statement"].item()
            t1 = data.loc[data["statementID"] == id1, "tag"].item()
            t2 = ar6_data.loc[ar6_data["statementID"] == id2, "tag"].item()
            prompt = chat_templates["llama3"].format(
                INSTRUCTION=ipcc_template.format(
                    CONTEXT1=c1,
                    CONTEXT2=c2,
                    STATEMENT1=s1,
                    STATEMENT2=s2
                ),
                ANSWER="Between Statement 1 and Statement 2, the more confident choice is Statement "
            )
            true = 0.5
            if tags.index(t1) > tags.index(t2): true = 1.
            if tags.index(t1) < tags.index(t2): true = 0.
            prompts.loc[len(prompts)] = [id1, id2, prompt, true]
    prompts.to_json(f"{data_storage}/climatex/prompts/AR{i}.jsonl", orient="records", lines=True)