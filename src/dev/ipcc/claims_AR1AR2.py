import os, sys, gc, re

HF_TOKEN = os.environ.get("HF_TOKEN")
from dev.constants import gdrive_path

from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import torch as t
import pandas as pd

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = t.device("cuda")

def free_mem(vars):
    for v in vars: del v
    gc.collect()
    t.cuda.empty_cache()


accelerator = Accelerator()
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct", 
                                             torch_dtype=t.float16, 
                                             device_map="auto",
                                             cache_dir="/gws/nopw/j04/ai4er/users/maiush/LLMs") 
model = accelerator.prepare(model); model.eval()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")


prompt_template = """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{INSTRUCTION1}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{RESPONSE1}<|eot_id|><|start_header_id|>user<|end_header_id|>

{INSTRUCTION2}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{RESPONSE2}<|eot_id|><|start_header_id|>user<|end_header_id|>

{INSTRUCTION}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{ANSWER}"""

system_prompt = """\
The assistant will read an abstract from a scientific report written in markdown, and return a list of claims extracted verbatim from this report. \
A claim is any synthesis of evidence, which may or may not be assigned a measure of uncertainty e.g., high confidence."""

instruction_template = """\
Read the following abstract and extract all scientific claims asserted. Print them verbatim, in a numbered list.

Abstract: {ABSTRACT}"""

# AR6 used for few-shot examples
ar6 = pd.read_json(f"{gdrive_path}/ipcc/long/long_parsed/AR6_claims.jsonl", orient="records", lines=True)
# cherry-picking good examples for few-shot
# some janky cleaning to include the formatting
abstract1 = ar6.at[6, "section"]
claims1 = [f"{claim.replace(' (_high confidence_)', '.')}" for i, claim in enumerate(ar6.at[6, 'claims'])]
claims1 = [c[:c.index(".")+1] for c in claims1]
claims1 = [f"{i+1}. {claim}" for i, claim in enumerate(claims1)]
claims1 = "\n".join(claims1)
abstract2 = ar6.at[17, "section"]
claims2 = [claim.replace(f" (_{tag}_)", ".") for claim, tag in zip(ar6.at[17, "claims"], ar6.at[17, "tags"])]
claims2 = [c.replace("**", "") for c in claims2]
claims2 = [c[:c.index(".")+1] for c in claims2]
claims2 = [f"{i+1}. {claim}" for i, claim in enumerate(claims2)]
claims2 = "\n".join(claims2)


file, type = sys.argv[1:3]
path = f"{gdrive_path}/ipcc/{type}/{type}_parsed/{file}.mmd"
with open(path, "r") as f: content = f.read()
sections = re.split(r'\n(?=#+ )', content)


answer_start = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
answer_end = "<|eot_id|>"
claims = pd.DataFrame(columns=["section", "claims"])
for section in sections:
    prompt = prompt_template.format(
        SYSTEM_PROMPT = system_prompt,
        INSTRUCTION1 = instruction_template.format(ABSTRACT = abstract1),
        RESPONSE1 = claims1,
        INSTRUCTION2 = instruction_template.format(ABSTRACT = abstract2),
        RESPONSE2 = claims2,
        INSTRUCTION = instruction_template.format(ABSTRACT = section),
        ANSWER = ""
    )
    tks = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    if tks.size(1) > 8192: continue
    with t.no_grad(): out = model.generate(inputs=tks, max_length=8192)
    answer = tokenizer.decode(out[0]) 
    ix_start = answer.rindex(answer_start)
    ix_end = answer.rindex(answer_end)
    row = [section, answer[ix_start+len(answer_start)+2:ix_end]]
    claims.loc[len(claims)] = row
outpath = f"{gdrive_path}/ipcc/{type}/{type}_parsed/{file}_claims.jsonl"
claims.to_json(outpath, orient="records", lines=True)