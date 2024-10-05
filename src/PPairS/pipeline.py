import gc
import torch as t
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from jaxtyping import Float
from typing import Union, List, Dict


def free_mem(vars):
    for v in vars: del v
    gc.collect()
    t.cuda.empty_cache()


class PPairSLMPipeline:

    def __init__(
            self,
            model: AutoModelForCausalLM,
            tokenizer: AutoTokenizer,
            mode: str
    ):
        self.model = model; self.model.eval()
        self.tokenizer = tokenizer
        assert mode in ["zero_shot", "compare", "contrast"]
        self.mode = mode
        if mode == "zero_shot":
            print(f"zero-shot. returning processed logits.")
            self.logit_ids = [tokenizer.encode(str(i), return_tensors="pt", add_special_tokens=False).flatten()[-1].item() for i in range(1, 6)]
            assert len(self.logit_ids) == 5
        elif mode == "compare":
            print(f"pairwise comparisons. returning processed logits.")
            self.logit_ids = [tokenizer.encode(str(i), return_tensors="pt", add_special_tokens=False).flatten()[-1].item() for i in [1, 2]]
            assert len(self.logit_ids) == 2
        elif mode == "contrast":
            print(f"contrast. returning harvested activations.")

    def __call__(
            self,
            messages: List[Dict[str, str]],
            verbose: bool=False,
            **kwargs
    ) -> Union[Float[Tensor, "1 n_vocab"], Float[Tensor, "d_model"]]:
        # apply chat template
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # if necessary, allow for continuation instead of QA
        prompt = self.check_continue(messages, prompt)
        if verbose: print(f"PROMPT\n{prompt}")
        # tokenize
        tks = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        # return logits or residual stream, depending on mode
        with t.inference_mode(): 
            if self.mode in ["zero_shot", "compare"]:
                out = self.model.generate(
                    tks.input_ids,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **kwargs
                )
                logits = t.stack(out["scores"]).squeeze(1).to(self.model.dtype)
                scores = logits[:, self.logit_ids]
                free_mem([tks, out, logits])
                return scores
            elif self.mode == "contrast":
                out = self.model(tks.input_ids, output_hidden_states=True)
                activations = out["hidden_states"][-1].squeeze(0)[-1, :]
                free_mem([tks, out])
                return activations

    def check_continue(
            self,
            messages: List[Dict[str, str]],
            prompt: str
    ) -> str:
        '''
        if we want continuation of the prompt instead of QA, we need to modify the prompt a bit.
        '''
        if messages[-1]["role"] != "assistant": return prompt
        message = messages[-1]["content"]
        space = message[-1] == " "
        if space: message = message[:-1]
        ix = prompt.rindex(message) + len(message)
        prompt = prompt[:ix]
        if space: prompt = prompt + " "
        return prompt