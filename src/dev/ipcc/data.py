from dev.constants import gdrive_path
import sys, re
import pandas as pd


long_path = f"{gdrive_path}/ipcc/long/long_parsed"
spm_path = f"{gdrive_path}/ipcc/summary/summary_parsed"

confidence_tags = [f"{level} confidence" for level in ["very low", "low", "medium", "high", "very high"]]
confidence_tags += [f"_{tag}_" for tag in confidence_tags] + [f"_({tag})_" for tag in confidence_tags] + [f"(_{tag}_)" for tag in confidence_tags]

def tag_root(tag: str) -> str:
    root = ""
    if "very" in tag: root += "very_"
    if "low" in tag: root += "low"
    if "medium" in tag: root += "medium"
    if "high" in tag: root += "high"
    return root

def clean_confidence(text: str) -> str:
    # find all sets of parentheses which include the word 'confidence' e.g., "warming has occurred (_high confidence_)"
    pattern_paren = r'\([^()]*\bconfidence(_)?\b[^()]*\)'
    # some confidence tags are expressed within the claim itself e.g., "there is _high confidence_ that"
    pattern_within = r'_[\w ]+confidence_ that'
    # continually remove confidence tags until done
    while True:
        cleaned = re.sub(pattern_paren, '', text, flags=re.IGNORECASE)
        cleaned = re.sub(pattern_within, 'evidence', cleaned, flags=re.IGNORECASE)
        if cleaned == text: break
        text = cleaned
    return text

file, type = sys.argv[1:3]
path = long_path if type == "long" else spm_path

data = pd.read_json(f"{path}/{file}_claims.jsonl", orient="records", lines=True)
ixs = []
for i, row in data.iterrows():
    section, claims, tags = row.values
    if len(claims) == 0: continue
    data.at[i, "section"] = clean_confidence(section)
    data.at[i, "claims"] = [clean_confidence(claim) for claim in claims]
    data.at[i, "tags"] = [tag_root(tag) for tag in tags]
    ixs.append(i)
data = data.loc[ixs]

processed = pd.DataFrame(columns=["contextID", "context", "statementID", "statement", "tag"])
c_id, s_id = 0, 0
done_c, done_s = [], []
for i, row in data.iterrows():
    context, statements, tags = row.values
    if context not in done_c:
        done_c.append(context)
        c_id += 1
    for statement, tag in zip(statements, tags):
        if statement not in done_s:
            done_s.append(statement)
            s_id += 1
        processed.loc[len(processed)] = [f"C{c_id}", context, f"S{s_id}", statement, tag]

outpath = f"{path}/{file}_processed.jsonl"
processed.to_json(outpath, orient="records", lines=True)