import os, sys, gc

HF_TOKEN = os.environ.get("HF_TOKEN")
from dev.constants import gdrive_path

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


accelerator = Accelerator()
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", 
                                             torch_dtype=t.float16, 
                                             device_map="auto",
                                             cache_dir="/gws/nopw/j04/ai4er/users/maiush/LLMs") 
model = accelerator.prepare(model); model.eval()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

file, ix = sys.argv[1:3]
# using the same prompts used in the ipcc experiment
inpath = f"{gdrive_path}/ipcc/long"
prompts = pd.read_json(f"{inpath}/prompts/{file}_prompts{ix}.jsonl", orient="records", lines=True)

outpath = f"{gdrive_path}/causal/activations/{file}_PART{ix}.pt"
if not os.path.exists(outpath):
    activations = t.zeros(len(prompts), d_model)
    for i in trange(len(prompts)):
        prompt = prompts.at[i, 'prompt']
        tks = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with t.inference_mode(): out = model(tks, output_hidden_states=True)
        activations[i] = out["hidden_states"][-1][0, -1, :].cpu()
        free_mem([tks, out])
    t.save(activations, outpath)
    free_mem([activations])