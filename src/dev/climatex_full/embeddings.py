import os, gc
HF_TOKEN = os.environ.get("HF_TOKEN")
from dev.constants import gdrive_path
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import pandas as pd
from tqdm import trange


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = t.device("cuda")

def free_mem(vars):
    for v in vars: del v
    gc.collect()
    t.cuda.empty_cache()
n_layer, d_model = 32, 4096


accelerator = Accelerator()
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", 
                                             torch_dtype=t.float16, 
                                             device_map="auto",
                                             cache_dir="/gws/nopw/j04/ai4er/users/maiush/LLMs") 
model = accelerator.prepare(model); model.eval()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

path = f"{gdrive_path}/climatex_full/claims"
files = os.listdir(path)

for file in files:
    claims = pd.read_json(f"{path}/{file}", orient="records", lines=True)
    if len(claims) == 0: continue

    prompts = claims["context"].tolist()
    activations = t.zeros(len(prompts), d_model)
    for i in trange(len(prompts), desc=file):
        tks = tokenizer.encode(prompts[i], return_tensors="pt", add_special_tokens=False).to(device)
        with t.inference_mode(): out = model(tks, output_hidden_states=True)
        activations[i] = out["hidden_states"][-1][0, -1, :].cpu()
        free_mem([tks, out])
    outpath = f"{gdrive_path}/climatex_full/embeddings/{file.split('_')[0]}.pt"
    t.save(activations, outpath)
    free_mem([activations])