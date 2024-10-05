import os, sys
from pathlib import Path
from tqdm import trange
import pandas as pd

HF_TOKEN = os.environ.get("HF_TOKEN")
from PPairS.constants import llm_cache, data_path, results_path
from PPairS.utils import models
from PPairS.pipeline import PPairSLMPipeline

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# main parameters
mode, model_name, dataset, aspect = sys.argv[1:5]
outpath = f"{results_path}/{dataset}/{model_name}"
Path(outpath).mkdir(exist_ok=True, parents=True)
outpath += f"/{aspect}_{mode}"
if mode == "contrast":
    choice = sys.argv[5]
    outpath += f"_{choice}"
item_names = {
    "newsroom": "summary",
    "summeval": "summary",
    "hanna": "story"
}
item_name = item_names[dataset]

# load data
if mode == "zero_shot":
    data = pd.read_json(f"{data_path}/{dataset}_prompts_zero_shot.jsonl", orient="records", lines=True)
else:
    data = pd.read_json(f"{data_path}/{dataset}_prompts_compare.jsonl", orient="records", lines=True)
# check for completed / partial runs
if os.path.exists(f"{outpath}.pt"):
    results = t.load(f"{outpath}.pt", weights_only=True)
    results = [x for x in results]
    if len(results) == len(data):
        print("results already exist")
        sys.exit(0)
else:
    results = []

# load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    models[model_name],
    torch_dtype=t.bfloat16,
    device_map="auto",
    cache_dir=llm_cache,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    models[model_name],
    cache_dir=llm_cache
)

# inference
pipeline = PPairSLMPipeline(model, tokenizer, mode)
for i in trange(len(results), len(data)):    
    if mode == "zero_shot":
        prompt = [
            {"role": "user", "content": data.at[i, aspect]},
            {"role": "assistant", "content": f"I would rate the {aspect} of this {item_name} as a "}
        ]
    elif mode == "compare":
        prompt = [
            {"role": "user", "content": data.at[i, aspect]},
            {"role": "assistant", "content": f"Between {item_name} 1 and {item_name} 2, the more {aspect} choice is {item_name} "}
        ]
    elif mode == "contrast":
        prompt = [
            {"role": "user", "content": data.at[i, aspect]},
            {"role": "assistant", "content": f"Between {item_name} 1 and {item_name} 2, the more {aspect} choice is {item_name} {choice}"}
        ]
    x = pipeline(prompt).squeeze()
    results.append(x.cpu())
t.save(t.stack(results, dim=0), f"{outpath}.pt")