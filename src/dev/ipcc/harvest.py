import os, sys, gc

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
n_layer, d_model = 32, 4096


model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", 
                                             torch_dtype=t.float16, 
                                             device_map="auto",
                                             cache_dir="/gws/nopw/j04/ai4er/users/maiush/LLMs") 
model.eval()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

long_path = f"{gdrive_path}/ipcc/long"
spm_path = f"{gdrive_path}/ipcc/summary"
file, type, ix, choice = sys.argv[1:5]
path = long_path if type == "long" else spm_path
prompts = pd.read_json(f"{path}/prompts/{file}_prompts{ix}.jsonl", orient="records", lines=True)

outpath = f"{path}/activations/{file}_PART{ix}_CHOICE{choice}.pt"
if not os.path.exists(outpath):
    activations = t.zeros(len(prompts), d_model)
    for i in trange(len(prompts)):
        prompt = f"{prompts.at[i, 'prompt']}{choice}"
        tks = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with t.no_grad(): out = model(tks, output_hidden_states=True)
        activations[i] = out["hidden_states"][-1][0, -1, :].cpu()
        free_mem([tks, out])
    t.save(activations, outpath)
    free_mem([activations])