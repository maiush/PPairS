import os, sys, gc
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN")
from dev.constants import gdrive_path

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
import pandas as pd
from tqdm import trange

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = t.device("cuda")

def free_mem(vars):
    for v in vars: del v
    gc.collect()
    t.cuda.empty_cache()

models = {
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-v0.1"
}
aspects = ["coherence", "consistency", "fluency", "relevance"]

model_name = sys.argv[1]
model = AutoModelForCausalLM.from_pretrained(models[model_name], torch_dtype=t.float16, device_map="auto"); model.eval()
tokenizer = AutoTokenizer.from_pretrained(models[model_name])
tk_scores = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 6)]
print(model_name)
dataset = sys.argv[2]
prompts = pd.read_json(f"{gdrive_path}/benchmarks/prompts/{model_name}/{dataset}_score.jsonl", orient="records", lines=True)
for aspect in aspects:
    outpath = f"{gdrive_path}/benchmarks/scores/{model_name}"
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if os.path.exists(f"{outpath}/{dataset}_{aspect}.jsonl"): continue

    scores = []
    for i in trange(len(prompts), desc=aspect):
        prompt = prompts.at[prompts.index[i], f"prompt_{aspect}"]
        tks = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with t.no_grad(): out = model(tks)
        logits = out["logits"][:, -1, tk_scores]
        score = t.argmax(logits[0]).item() + 1
        scores.append(score)
        free_mem([tks, out, logits])
    prompts["score"] = scores
    prompts.to_json(f"{outpath}/{dataset}_{aspect}.jsonl", orient="records", lines=True)

free_mem([model, tokenizer, tk_scores])