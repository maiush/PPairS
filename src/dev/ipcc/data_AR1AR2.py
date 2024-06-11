import sys, re
from dev.constants import gdrive_path
import pandas as pd


file = sys.argv[1]
path = f"{gdrive_path}/ipcc/long/long_parsed"
data = pd.read_json(f"{path}/{file}_claims.jsonl", orient="records", lines=True)

processed = pd.DataFrame(columns=["contextID", "context", "statementID", "statement"])
c_id, s_id = 0, 0
done_c, done_s = [], []
for i, row in data.iterrows():
    context, answer = row.values
    lines = answer.split("\n")
    claims = [l for l in lines if re.match(r'^\d+\. ', l)]
    claims = [c[c.index(". ")+2:] for c in claims]
    if len(claims) == 0: continue
    if context not in done_c:
        done_c.append(context)
        c_id += 1
    for statement in claims:
        if statement not in done_s:
            done_s.append(statement)
            s_id += 1
        processed.loc[len(processed)] = [f"C{c_id}", context, f"S{s_id}", statement]

outpath = f"{path}/{file}_processed.jsonl"
processed.to_json(outpath, orient="records", lines=True)