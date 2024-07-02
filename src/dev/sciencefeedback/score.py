import os, sys, gc
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN")
from dev.constants import data_storage

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
import pandas as pd

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = t.device("cuda")

def free_mem(vars):
    for v in vars: del v
    gc.collect()
    t.cuda.empty_cache()


model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=t.float16, device_map="auto"); model.eval()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

answers = ["incorrect", "misleading", "correct"]
tk_answers = [tokenizer.encode(x, add_special_tokens=False)[0] for x in answers]

topic = sys.argv[1]
prompts = pd.read_json(f"{data_storage}/sciencefeedback/prompts/{topic}feedback_score.jsonl", orient="records", lines=True)
outpath = f"{data_storage}/sciencefeedback/scores"
Path(outpath).mkdir(parents=True, exist_ok=True)
if not os.path.exists(f"{outpath}/{topic}feedback.jsonl"):
    verdicts = []
    for i in range(len(prompts)):
        prompt = prompts.at[i, "prompt"]
        tks = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with t.no_grad(): out = model(tks)
        logits = out["logits"][:, -1, tk_answers]
        ans = answers[t.argmax(logits[0]).item()]
        verdicts.append(ans)
        free_mem([tks, out, logits])
    prompts["verdict"] = verdicts
    prompts.to_json(f"{outpath}/{topic}feedback.jsonl", orient="records", lines=True)