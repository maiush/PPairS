import os, gc
HF_TOKEN = os.environ.get("HF_TOKEN")
from dev.constants import gdrive_path
from PPairS.prompts import chat_templates, ipcc_template
from PPairS.sort import mergesort
from PPairS.classify import CompareClassifier

import torch as t
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

import numpy as np
import pandas as pd

from jaxtyping import Float
from typing import Tuple, Optional
from tqdm import tqdm


class InsertIPCC:

    def __init__(
            self,
            llm_cache: str="/gws/nopw/j04/ai4er/users/maiush/LLMs",
            clf_path: str="/gws/nopw/j04/ai4er/users/maiush/PPairS/src/dev/ipcc_binary_search/clf_100.pt",
            device: str="cuda"
    ):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        self.device = t.device(device)
        self.n_layer, self.d_model = 32, 4096

        accelerator = Accelerator()
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", 
                                                    torch_dtype=t.float16, 
                                                    device_map="auto",
                                                    cache_dir=llm_cache) 
        self.model = accelerator.prepare(self.model); self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

        file_path = f"{gdrive_path}/ipcc/long/long_parsed"
        self.claims = pd.read_json(f"{file_path}/AR6_processed.jsonl", orient="records", lines=True)

        self.sort_AR6()
        self.load_classifier(clf_path)

    def sort_AR6(self):
        file_path = f"{gdrive_path}/ipcc/long/prompts"
        files = [f for f in os.listdir(file_path) if "AR6" in f]

        P, ids = {}, []
        for file in tqdm(files, desc="ground truth: AR6"):
            df = pd.read_json(f"{file_path}/{file}", orient="records", lines=True)
            for _, row in df.iterrows():
                s1 = row["S1"]; s2 = row["S2"]; p = row["P(S1)"]
                if s1 not in ids:
                    # add s1 to each existing key
                    for id in ids: P[id][s1] = np.nan
                    # add s1 as a new key itself
                    P[s1] = {s1: 1.}
                    ids.append(s1)
                if s2 not in ids:
                    # add s2 to each existing key
                    for id in ids: P[id][s2] = np.nan
                    # add s2 as a new key itself
                    P[s2] = {s2: 1.}
                    ids.append(s2)
                P[s1][s2] = p
                P[s2][s1] = 1-p
        P = pd.DataFrame(P)
        ixs = [i for i in range(len(ids))]
        sorted_ixs = mergesort(
            ixs=ixs,
            P=P.to_numpy(),
            beam=False
        )
        self.unordered_ixs = P.columns.to_numpy()
        self.ordered_ixs = P.columns[sorted_ixs].to_numpy()

    def load_classifier(self, path: str):
        self.clf = CompareClassifier(d_model=self.d_model, device=self.device)
        self.clf.load_state_dict(t.load(path, map_location=self.device))
    
    def insert(self, context: str, statement: str, low: Optional[int]=None, high: Optional[int]=None) -> Tuple[Optional[str]]:
        '''
        return a tuple of the left and right statementID's (between which we'd insert the new statement)
        at an edge we return None on the appropriate side (left or right)
        '''
        if low is None: low = 0
        if high is None: high = len(self.ordered_ixs)-1

        if high >= low:
            mid = (high + low) // 2
            comparison = self.test_statement(context, statement, self.ordered_ixs[mid])
            # note: ixs is sorted in descending order
            if comparison == 1.:
                # statement is left (higher confidence) of mid
                if mid == low:
                    leftID = self.ordered_ixs[low-1] if low != 0 else None
                    rightID = self.ordered_ixs[low]
                    return (leftID, rightID)
                else: return self.insert(context, statement, low, mid-1)
            else:
                # P(S1) == 0 and statement is right (lower confidence) of mid
                if mid == high:
                    leftID = self.ordered_ixs[high]
                    rightID = self.ordered_ixs[high+1] if high != len(self.ordered_ixs)-1 else None
                    return (leftID, rightID)
                else: return self.insert(context, statement, mid+1, high)
        else:
            # this should never happen! if it does, something went wrong
            return -1.

    def test_statement(self, context: str, statement: str, statementID: str) -> float:
        context2, statement2 = self.get_context_and_statement(statementID)
        prompt = self.format_prompt(
            context1=context,
            context2=context2,
            statement1=statement,
            statement2=statement2
        )
        x_pos = self.harvest(prompt=prompt, choice=1)
        x_neg = self.harvest(prompt=prompt, choice=2)
        x = x_pos - x_neg
        label = self.classify(x)
        return label
    
    def classify(self, x: Float[Tensor, "d_model"]) -> float:
        with t.inference_mode(): logit = self.clf(x)
        p = F.sigmoid(logit)
        if p > 0.5: return 1.
        else: return 0.

    def harvest(self, prompt: str, choice: int) -> Float[Tensor, "d_model"]:
        tks = self.tokenizer.encode(f"{prompt}{choice}", return_tensors="pt", add_special_tokens=False).to(self.device)
        with t.inference_mode(): out = self.model(tks, output_hidden_states=True)
        acts = out["hidden_states"][-1][0, -1, :].cpu()
        del tks, out
        gc.collect()
        t.cuda.empty_cache()
        return acts

    def format_prompt(self, context1: str, context2: str, statement1: str, statement2: str) -> str:
        prompt = chat_templates["llama3"].format(
            INSTRUCTION=ipcc_template.format(
                CONTEXT1=context1,
                CONTEXT2=context2,
                STATEMENT1=statement1,
                STATEMENT2=statement2
            ),
            ANSWER="Between Statement 1 and Statement 2, the more confident choice is Statement "
        )
        return prompt
    
    def get_context_and_statement(self, statementID: str) -> Tuple[str]:
        row = self.claims.loc[self.claims["statementID"] == statementID, ["context", "statement"]]
        assert len(row) == 1
        c, s = row.iloc[0].tolist()
        return (c, s)