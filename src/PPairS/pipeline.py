import torch as t
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from jaxtyping import Float
from typing import Union, List, Dict


class PPairSLMPipeline:

    def __init__(
            self,
            model: AutoModelForCausalLM,
            tokenizer: AutoTokenizer,
            mode: str
    ):
        self.model = model
        self.tokenizer = tokenizer
        assert mode in ["zero-shot", "contrast"]
        self.mode = mode
        if mode == "zero-shot":
            print(f"zero-shot. returning processed logits.")
        elif mode == "contrast":
            print(f"contrast. returning harvested activations.")

    def __call__(
            self,
            messages: List[Dict[str, str]],
            verbose: bool=False,
            **kwargs
    ) -> Union[Float[Tensor, "n_seq n_vocab"], Float[Tensor, "d_model"]]:
        # apply chat template
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # if necessary, allow for continuation instead of QA
        prompt = self.check_continue(messages, prompt)
        if verbose: print(f"PROMPT\n{prompt}")
        # tokenize
        tks = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        # return logits or residual stream, depending on mode
        with t.inference_mode(): 
            if self.mode == "zero-shot":
                # default value for max_new_tokens
                max_new_tokens = kwargs.pop("max_new_tokens", 128)
                out = self.model.generate(
                    tks.input_ids,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **kwargs
                )
                return t.stack(out["scores"]).squeeze(1).to(self.model.dtype)
            elif self.mode == "contrast":
                out = self.model(tks.input_ids, output_hidden_states=True)
                return out["hidden_states"][-1].squeeze(0)[-1, :]

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
        ix = prompt.rindex(message) + len(message)
        return prompt[:ix]