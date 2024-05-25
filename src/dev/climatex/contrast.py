import os, sys, gc
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN")
from dev.constants import gdrive_path

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
import torch.nn.functional as F
import pandas as pd

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = t.device("cuda")

def free_mem(vars):
    for v in vars: del v
    gc.collect()
    t.cuda.empty_cache()
n_layer, d_model = 32, 4096


model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=t.float16, device_map="auto"); model.eval()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

choice = sys.argv[1]
prompts = pd.read_json(f"{gdrive_path}/climatex/prompts_{choice}.jsonl", orient="records", lines=True)
outpath = f"{gdrive_path}/climatex/activations"
Path(outpath).mkdir(parents=True, exist_ok=True)
if not os.path.exists(f"{outpath}/{choice}.pt"):
    activations = t.zeros(len(prompts), d_model)
    for i in range(len(prompts)):
        prompt = prompts.at[i, f"prompt"]
        tks = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with t.no_grad(): out = model(tks, output_hidden_states=True)
        activations[i] = out["hidden_states"][-1][0, -1, :].cpu()
        free_mem([tks, out])
    t.save(activations, f"{outpath}/{choice}.pt")
    free_mem([activations])