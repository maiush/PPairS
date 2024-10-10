import os
from pathlib import Path
from tqdm import trange
from typing import Optional, Tuple, List, Union
from jaxtyping import Float
import pandas as pd

HF_TOKEN = os.environ.get('HF_TOKEN')
from PPairS.constants import llm_cache, results_path
from PPairS.utils import models
from PPairS.pipeline import PPairSLMPipeline

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
from torch import Tensor

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def load_model_and_tokenizer(model: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        models[model],
        torch_dtype=t.bfloat16,
        device_map="auto",
        cache_dir=llm_cache,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        models[model],
        cache_dir=llm_cache
    )
    return model, tokenizer

def run_pipeline(
        outpath: str,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        mode: str,
        dataset: PPairSDataset,
        data: pd.DataFrame,
        results: Optional[List[Union[Float[Tensor, '1 n_vocab'], Float[Tensor, 'd_model']]]]=None
) -> None:
    pipeline = PPairSLMPipeline(model, tokenizer, mode)
    for i in trange(len(results), dataset.length):
        prompt = dataset.get_prompt(i)
        x = pipeline(prompt).squeeze()
        results.append(x.cpu())
        t.save(t.stack(results, dim=0), f'{outpath}.pt')

def inference(
        mode: str,
        model: str,
        dataset: str,
        aspect: Optional[str]=None,
        choice: Optional[str]=None
) -> None:
    # results directory path
    outpath = f'{results_path}/{dataset}/{model}'
    Path(outpath).mkdir(exist_ok=True, parents=True)
    # results file path
    if aspect is not None: outpath += f'/{aspect}_'
    outpath += f'{mode}'
    if mode == 'contrast': outpath += f'_{choice}'
    # load dataset
    dataset = PPairSDataset(dataset, mode=mode, aspect=aspect, choice=choice)
    # check for completed / partial runs
    if os.path.exists(f'{outpath}.pt'):
        results = t.load(f'{outpath}.pt', weights_only=True)
        results = [x for x in results]
        if len(results) == dataset.length:
            print('results already exist')
            return
    else: results = []
    # load model and tokenizer
    m, t = load_model_and_tokenizer(model)
    # run pipeline
    run_pipeline(
        outpath,
        m,
        t,
        mode,
        dataset,
        results
    )
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, choices=['zero_shot', 'compare', 'contrast'], required=True)
    parser.add_argument('-model', type=str, choices=list(models.keys()), required=True)
    parser.add_argument('-dataset', type=str, choices=['newsroom', 'summeval', 'hanna', 'rocstories'], required=True)
    parser.add_argument('-aspect', type=str, required=False)
    parser.add_argument('-choice', type=str, required=False)
    args = parser.parse_args()

    # argument checks
    if args.dataset in ['newsroom', 'summeval', 'hanna']:
        assert args.aspect is not None
    if args.mode == 'contrast':
        assert args.choice is not None

    inference(*args)