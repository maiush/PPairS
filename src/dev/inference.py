import os, sys, pickle
from pathlib import Path
from tqdm import trange
import pandas as pd

HF_TOKEN = os.environ.get("HF_TOKEN")
from dev.constants import llm_cache, data_path, results_path
from PPairS.utils import models
from PPairS.pipeline import PPairSLMPipeline

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# main parameters
mode = sys.argv[1]
assert mode in ["zero_shot", "compare", "contrast"]
model_name, dataset, aspect = sys.argv[2:5]
outpath = f"{results_path}/{dataset}/{model_name}"
Path(outpath).mkdir(exist_ok=True, parents=True)
outpath += f"/{aspect}_{mode}"
if mode == "contrast":
    answer_template = """Between {ITEM} 1 and {ITEM} 2, the more {ASPECT} choice is {ITEM} """
    choice = sys.argv[5]
    outpath += f"_{choice}"
if os.path.exists(f"{outpath}.pt"):
    print("results already exist")
    sys.exit(0)

max_new_tokens=128
item_names = {
    "newsroom": "summary",
    "summeval": "summary",
    "hanna": "story"
}
item_name = item_names[dataset]


# load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    models[model_name],
    torch_dtype=t.bfloat16,
    device_map="auto",
    cache_dir=llm_cache,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    models[model_name],
    cache_dir=llm_cache
)


# inference
pipeline = PPairSLMPipeline(model, tokenizer, mode)
results, answers = [], []
if mode == "zero_shot":
    data = pd.read_json(f"{data_path}/{dataset}_prompts_zero_shot.jsonl", orient="records", lines=True)
    for i in trange(len(data)):
        prompt = [
            {"role": "user", "content": data.at[i, aspect]}
        ]
        answer, x = pipeline(prompt, verbose=False, max_new_tokens=max_new_tokens)
        x = F.pad(x, (0, 0, 0, max_new_tokens-len(x)), mode="constant", value=-1)
        results.append(x.cpu())
        answers.append(answer)
elif mode == "compare":
    data = pd.read_json(f"{data_path}/{dataset}_prompts_compare.jsonl", orient="records", lines=True)
    for i in trange(len(data)):
        prompt = [
            {"role": "user", "content": data.at[i, aspect]}
        ]
        answer, x = pipeline(prompt, verbose=False, max_new_tokens=max_new_tokens)
        x = F.pad(x, (0, 0, 0, max_new_tokens-len(x)), mode="constant", value=-1)
        results.append(x.cpu())
        answers.append(answer)
elif mode == "contrast":
    data = pd.read_json(f"{data_path}/{dataset}_prompts_compare.jsonl", orient="records", lines=True)
    for i in trange(len(data)):
        prompt = [
            {"role": "user", "content": data.at[i, aspect]},
            {"role": "assistant", "content": f"{answer_template.format(ITEM=item_name, ASPECT=aspect)}{choice}"}
        ]
        x = pipeline(prompt)
        results.append(x.cpu())


# save results
results = t.stack(results, dim=0)
t.save(results, f"{outpath}.pt")
# save answers if possible
if len(answers) > 0:
    with open(f"{outpath}_answers.pkl", "wb") as outfile: pickle.dump(answers, outfile)