import os, sys, gc
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN")
from dev.constants import data_storage

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
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1"
}
aspects = ["coherence", "consistency", "fluency", "relevance"]

model_name = sys.argv[1]
model = AutoModelForCausalLM.from_pretrained(models[model_name], torch_dtype=t.float16, device_map="auto"); model.eval()
tokenizer = AutoTokenizer.from_pretrained(models[model_name])
special_char = "Ġ" if model_name == "llama3" else "▁"
tks_A = tokenizer.convert_tokens_to_ids(["A", f"{special_char}A"])
tks_B = tokenizer.convert_tokens_to_ids(["B", f"{special_char}B"])

dataset = sys.argv[2]
prompts = pd.read_json(f"{data_storage}/benchmarks/prompts_short/{model_name}/{dataset}_theirs.jsonl", orient="records", lines=True)
for aspect in aspects:
    outpath = f"{data_storage}/benchmarks/logits/{model_name}"
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if os.path.exists(f"{outpath}/{dataset}_{aspect}.jsonl"): continue

    p_s1, p_s2 = [], []
    for i in trange(len(prompts), desc=f"{dataset}:{aspect}"):
        prompt = prompts.at[i, f"prompt_{aspect}"]
        tks = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with t.no_grad():
            out = model.generate(
                inputs=tks,
                return_dict_in_generate=True,
                output_logits=True,
                max_new_tokens=32,
                do_sample=False,
                temperature=None,
                top_p=None
            )
        for ix, tk in enumerate(out["sequences"].squeeze(0)[-len(out["logits"]):]):
            if tk.item() in tks_A + tks_B:
                P = F.softmax(out["logits"][ix], dim=-1)
                p_s1.append(P[:, tks_A].sum().item())
                p_s2.append(P[:, tks_B].sum().item())
                break
        else:
            p_s1.append(0.5)
            p_s2.append(0.5)
        free_mem([tks, out])
    prompts["p_s1"] = p_s1
    prompts["p_s2"] = p_s2
    
    prompts.to_json(f"{outpath}/{dataset}_{aspect}.jsonl", orient="records", lines=True)