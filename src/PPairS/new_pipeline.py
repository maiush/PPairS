import gc
import torch as t
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from jaxtyping import Float
from typing import Optional, Union, List, Dict, Iterable


def free_mem(vars):
    for v in vars: del v
    gc.collect()
    t.cuda.empty_cache()


class PPairSLMPipeline:

    def __init__(
            self,
            model: AutoModelForCausalLM,
            tokenizer: AutoTokenizer,
            mode: str,
            zero_shot_options: Optional[Iterable[str]]=None,
            compare_options: Optional[Iterable[str]]=None
    ) -> None:
        self.model = model; self.model.eval()
        self.tokenizer = tokenizer
        self.mode = mode
        assert mode in ['zero_shot', 'compare', 'contrast']
        if mode == 'zero_shot':
            print(f'zero-shot. returning processed logits.')
            assert zero_shot_options is not None
            self.logit_ids = [tokenizer.encode(option, return_tensors="pt", add_special_tokens=False).flatten()[-1].item() for option in zero_shot_options]
            assert len(self.logit_ids) == len(zero_shot_options)
        elif mode == 'compare':
            print(f'pairwise comparison. returning processed logits.')
            assert compare_options is not None
            self.logit_ids = [tokenizer.encode(option, return_tensors="pt", add_special_tokens=False).flatten()[-1].item() for option in compare_options]
            assert len(self.logit_ids) == len(compare_options)
        elif mode == 'contrast':
            print(f'contrast. returning harvested activations.')

    def __call__(
            self,
            messages: List[Dict[str, str]],
            **kwargs
    ) -> Union[Float[Tensor, '1 n_vocab'], Float[Tensor, 'd_model']]:
        # apply chat template
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # if necessary, allow for continuation instead of QA
        prompt = self.check_continue(messages, prompt)
        # tokenize
        tks = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        # return logits or residual stream, depending on mode
        with t.inference_mode(): 
            if self.mode in ['zero_shot', 'compare']:
                out = self.model.generate(
                    tks.input_ids,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **kwargs
                )
                # return scores (these differ slightly from the actual logits - they've been processed a bit)
                logits = t.stack(out['scores']).squeeze(1).to(self.model.dtype)
                scores = logits[:, self.logit_ids]
                free_mem([tks, out, logits])
                return scores
            elif self.mode == 'contrast':
                out = self.model(tks.input_ids, output_hidden_states=True)
                # grab the residual stream after the last block
                activations = out['hidden_states'][-1].squeeze(0)[-1, :]
                free_mem([tks, out])
                return activations

    def check_continue(
            self,
            messages: List[Dict[str, str]],
            prompt: str
    ) -> str:
        '''
        if we want continuation of the prompt instead of QA, we need to modify it a bit.
        '''
        # this only applies if we're forcing the assistant to say something and then continue
        if messages[-1]['role'] != 'assistant': return prompt
        message = messages[-1]['content']
        # we need to handle the case where the last character is a space
        space = message[-1] == ' '
        if space: message = message[:-1]
        # we need to chop off the chat template tags added by the tokenizer
        ix = prompt.rindex(message) + len(message)
        prompt = prompt[:ix]
        # add the space back in necessary
        if space: prompt = prompt + " "
        return prompt