import sys, re
from dev.constants import gdrive_path
import pandas as pd
from typing import List
from tqdm import tqdm


def get_chunks(path: str) -> List[str]:
    chunks = []
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
        lines = text.split("\n")
        chunk = ""
        for line in lines:
            if line.startswith("#"):
                if chunk: chunks.append(chunk.strip())
                chunk = line + "\n"
            else:
                chunk += line + "\n"
        if chunk: chunks.append(chunk.strip())
    return chunks

long_path = f"{gdrive_path}/ipcc/long/long_parsed"
spm_path = f"{gdrive_path}/ipcc/summary/summary_parsed"
# we will loop through all confidence tags, searching for them within the text
# therefore: order is important!
confidence_tags = []
for tag in ["very high", "very low", "high", "low", "medium"]:
    confidence_tags.extend([f"{tag} confidence", f"_{tag} confidence_", f"_({tag} confidence)_", f"(_{tag} confidence_)"])

file, type = sys.argv[1:3]
path = long_path if type == "long" else spm_path
chunks = get_chunks(f"{path}/{file}.mmd")
df = pd.DataFrame(columns=["section", "claims", "tags"])
for chunk in chunks:
    sentences = re.split(r'[.!?]\s+|\n', chunk)
    claims, tags = [], []
    for sentence in sentences:
        if len(sentence) == 0: continue
        if sentence in confidence_tags: continue
        
        for tag in confidence_tags:
            # scan each sentence for a confidence tag
            if re.search(tag, sentence, re.IGNORECASE):
                claims.append(sentence.strip())
                tags.append(tag)
                break
            # some confidence tags are included *after* the given sentence
            extra_checks = [f"{sentence}{p} {tag}" for p in [".", "!", "?"]]
            broken = False
            for check in extra_checks:
                if check.lower() in chunk.lower():
                    claims.append(check.strip())
                    tags.append(tag)
                    broken = True; break
            if broken: break
    df.loc[len(df)] = [chunk, claims, tags]
outpath = f"{path}/{file}_claims.jsonl"
df.to_json(outpath, orient="records", lines=True)