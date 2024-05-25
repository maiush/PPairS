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


model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=t.float16, device_map="auto"); model.eval()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tks_1 = tokenizer.convert_tokens_to_ids(["1", "Ġ1"])
tks_2 = tokenizer.convert_tokens_to_ids(["2", "Ġ2"])

topic = sys.argv[1]
prompts = pd.read_json(f"{gdrive_path}/sciencefeedback/prompts/{topic}feedback_compare.jsonl", orient="records", lines=True)

outpath = f"{gdrive_path}/sciencefeedback/logits"
Path(outpath).mkdir(parents=True, exist_ok=True)
if not os.path.exists(f"{outpath}/{topic}feedback.jsonl"):
    p_s1, p_s2 = [], []
    for i in range(len(prompts)):
        prompt = prompts.at[i, f"prompt"]
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
            if tk.item() in tks_1 + tks_2:
                P = F.softmax(out["logits"][ix], dim=-1)
                p_s1.append(P[:, tks_1].sum().item())
                p_s2.append(P[:, tks_2].sum().item())
                break
        else:
            p_s1.append(0.5)
            p_s2.append(0.5)
        free_mem([tks, out])
    prompts["p_s1"] = p_s1
    prompts["p_s2"] = p_s2

    prompts.to_json(f"{outpath}/{topic}feedback.jsonl", orient="records", lines=True)