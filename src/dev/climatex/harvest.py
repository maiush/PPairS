import os, sys, gc
HF_TOKEN = os.environ.get("HF_TOKEN")

from dev.constants import data_storage
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
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


i = int(sys.argv[1])
prompts = pd.read_json(f"{data_storage}/climatex/prompts/AR{i}.jsonl", orient="records", lines=True)

# load llm
accelerator = Accelerator()
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", 
                                             torch_dtype=t.float16, 
                                             device_map="auto",
                                             cache_dir="/gws/nopw/j04/ai4er/users/maiush/LLMs") 
model = accelerator.prepare(model); model.eval()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# harvest activations
choice = int(sys.argv[2])
outpath = f"{data_storage}/climatex/activations/AR{i}_{choice}.pt"
if not os.path.exists(outpath):
    activations = t.zeros(len(prompts), d_model)
    for i in trange(len(prompts)):
        pair = []
        prompt = f"{prompts.at[i, "prompt"]}{choice}"
        tks = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
        with t.inference_mode(): out = model(tks, output_hidden_states=True)            
        activations[i] = out["hidden_states"][-1][0, -1, :].cpu()
        free_mem([tks, out])
    t.save(activations, outpath)
    free_mem([activations])