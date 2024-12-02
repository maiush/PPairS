import os, subprocess, pickle
from pathlib import Path

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from PPairS.constants import llm_cache, peft_path
from PPairS.utils import models, free_mem
from PPairS.data import PPairSDataset, PPairSPEFTDataset
from PPairS.pipeline import PPairSPEFTPipeline
from PPairS.inference import run_pipeline

from typing import Union, Optional, Tuple

HF_TOKEN = os.environ.get('HF_TOKEN')


def load_model_and_tokenizer(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    use_cache = False if model_name.startswith('gemma') else True
    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        models[model_name],
        torch_dtype=t.bfloat16,
        device_map="auto",
        cache_dir=llm_cache,
        trust_remote_code=True,
        use_cache=use_cache
    )
    tokenizer = AutoTokenizer.from_pretrained(
        models[model_name],
        cache_dir=llm_cache
    )
    # try gradient checkpointing and flash-attn
    try: model.gradient_checkpointing_enable() 
    except: print('could not enable gradient checkpointing')
    try: model.config.use_flash_attention = True    
    except: print('could not enable flash attn')
    return model, tokenizer

def train_lora(
        outpath: str,
        model: str,
        dataset: str,
        mode: str,
        rank: int,
        alpha: int,
        dropout: float,
        epoch: int,
        lr: float,
        split: Union[int, float],
        aspect: Optional[str]=None,
        reversed: Optional[str]=None
) -> None:
    # check for existing results
    checkpoints = [f for f in os.listdir(outpath) if f.startswith('checkpoint')]
    if len(checkpoints) > 0:
        if os.path.exists(f'{outpath}/eval.pt'):
            print(f'existing results: remove if you wish to retrain')
            return
        else:
            # we have partially completed results and need to start again
            command = f'rm -rf {outpath}'
            subprocess.run(command, shell=True)
            Path(outpath).mkdir(exist_ok=True, parents=True)        
    reversed = True if reversed == 'True' else False
    # load prompt dataset
    data = PPairSDataset(
        name=dataset,
        mode=mode,
        split=split,
        aspect=aspect,
        reversed=reversed,
        peft=True
    )
    # load model and tokenizer
    mod, tok = load_model_and_tokenizer(model)
    # prepare for peft
    train_data = PPairSPEFTDataset(data, tok)
    # prepare pipeline for training
    pipeline = PPairSPEFTPipeline(mod, rank=rank, alpha=alpha, dropout=dropout)
    # fit lora
    print('training lora')
    pipeline.train(
        train_data, 
        train_data,
        output_dir=outpath,
        n_epoch=epoch,
        lr=lr
    )
    # cleanup, before running evaluation
    free_mem([data, train_data, pipeline])
    # apply lora
    print('evaluating')
    checkpoint = [f for f in os.listdir(outpath) if f.startswith('checkpoint')][0]
    mod = PeftModel.from_pretrained(
        model=mod,
        model_id=f'{outpath}/{checkpoint}',
        device_map='auto',
        is_trainable=False
    )
    # run evaluation
    if mode == 'compare':
        for reversed in [True, False]:
            eval_path = f'{outpath}/eval'
            if reversed: eval_path += '_reversed'
            data = PPairSDataset(dataset, mode=mode, aspect=aspect, reversed=reversed)
            run_pipeline(eval_path, mod, tok, mode, data, [])
    else:
        # zero shot
        data = PPairSDataset(dataset, mode=mode, aspect=aspect)
        run_pipeline(f'{outpath}/eval', mod, tok, mode, data, [])
    print('done')


if __name__ == '__main__':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    import argparse, hashlib
    def parse_split(value: Union[int, float]) -> Union[int, float]:
        try:
            float_val = float(value)
            if float_val < 0: raise argparse.ArgumentTypeError(f'{value} must be positive')
            if float_val > 1 and float_val.is_integer(): return int(float_val)
            elif 0 <= float_val <= 1: return float_val
            else: raise argparse.ArgumentTypeError(f'{value} must be either an integer > 1 or float between 0 and 1')
        except: raise argparse.ArgumentTypeError(f'{value} is not a valid number')

    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, required=True, choices=list(models.keys()))
    parser.add_argument('-dataset', type=str, required=True, choices=[
        'newsroom',
        'summeval',
        'hanna',
        'rocstories',
        'caters',
        'mctaco'
    ])
    parser.add_argument('-mode', type=str, required=True, choices=['zero_shot', 'compare'])
    parser.add_argument('-aspect', type=str, required=False)
    parser.add_argument('-reversed', type=str, required=False)
    parser.add_argument('-split', type=parse_split, required=False, default=0.5)
    parser.add_argument('-rank', type=int, required=False, default=8)
    parser.add_argument('-alpha', type=int, required=False, default=16)
    parser.add_argument('-dropout', type=float, required=False, default=0.1)
    parser.add_argument('-epoch', type=int, required=False, default=3)
    parser.add_argument('-lr', type=float, required=False, default=0.0002)
    args = parser.parse_args()

    # argument checks
    if args.dataset in ['newsroom', 'summeval', 'hanna']:
        assert args.aspect is not None

    id = hashlib.sha1(str(args).encode('utf-8')).hexdigest()
    print(f'run id: {id}')
    outpath = f'{peft_path}/{id}'
    Path(outpath).mkdir(exist_ok=True, parents=True)
    with open(f'{outpath}/config.pkl', 'wb') as config_file:
        pickle.dump(args.__dict__, config_file)
    train_lora(outpath=outpath, **args.__dict__)