import os, sys, gc
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN")
from dev.constants import gdrive_path

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
import torch.nn.functional as F
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
n_layer, d_model = 32, 4096
aspects = ["coherence", "consistency", "fluency", "relevance"]

model_name = sys.argv[1]
model = AutoModelForCausalLM.from_pretrained(models[model_name], torch_dtype=t.float16, device_map="auto"); model.eval()
tokenizer = AutoTokenizer.from_pretrained(models[model_name])

dataset = sys.argv[2]
choice = sys.argv[3]
path = f"{gdrive_path}/benchmarks/prompts_short/{model_name}/{dataset}_mine_{choice}.jsonl"
prompts = pd.read_json(path, orient="records", lines=True)
for aspect in aspects:
    outpath = f"{gdrive_path}/benchmarks/activations/{model_name}"
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if os.path.exists(f"{outpath}/{dataset}_{aspect}_{choice}.pt"): continue

    activations = t.zeros(len(prompts), d_model)
    for i in trange(len(prompts), desc=f"{dataset}:{aspect}:{choice}"):
        prompt = prompts.at[i, f"prompt_{aspect}"]
        tks = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with t.no_grad(): out = model(tks, output_hidden_states=True)
        activations[i] = out["hidden_states"][-1][0, -1, :].cpu()
        free_mem([tks, out])
    t.save(activations, f"{outpath}/{dataset}_{aspect}_{choice}.pt")
    free_mem([activations])