import pandas as pd

from torch import Tensor
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from PPairS.constants import data_path

from typing import Optional, List, Dict, Union
from tqdm import trange


class PPairSDataset:

    items = {
        'newsroom': 'summary',
        'summeval': 'summary',
        'hanna': 'story',
        'rocstories': 'answer'
    }

    aspects_noun2adj = {
        'informativeness': 'informative',
        'relevance': 'relevant',
        'fluency': 'fluent',
        'coherence': 'coherent',
        'consistency': 'consistent',
        'empathy': 'empathetic',
        'surprise': 'surprising',
        'engagement': 'engaging',
        'complexity': 'complex'
    }

    scoring_datasets = ['newsroom', 'summeval', 'hanna']
    comparison_datasets = ['rocstories']
    grounding_datasets = ['caters', 'mctaco']
    all_datasets = scoring_datasets + comparison_datasets + grounding_datasets

    def __init__(
            self,
            name: str,
            mode: str,
            split: Optional[Union[int, float]]=None,
            aspect: Optional[str]=None,
            choice: Optional[str]=None,
            reversed: Optional[Union[bool, str]]=None,
            peft: Optional[bool]=False
    ) -> None:
        assert name in self.all_datasets
        self.name = name
        if name in self.scoring_datasets:
            assert aspect is not None
        if name == 'rocstories':
            aspect = 'consistency'
        self.aspect = aspect
        assert mode in ['zero_shot', 'compare', 'contrast']
        self.mode = mode
        if mode == 'contrast':
            assert choice is not None
        self.choice = choice
        # load prompts
        prompts_path = f'{data_path}/{name}_prompts_'
        if mode == 'zero_shot': 
            prompts_path += 'zero_shot'
        else: 
            prompts_path += 'compare'
            if isinstance(reversed, str): reversed = eval(reversed)
            if reversed: prompts_path += '_reversed' if reversed else ''
        prompts_path += '.jsonl'
        self.data = pd.read_json(prompts_path, orient='records', lines=True)
        if split is not None:
            if 0 <= split <= 1: self.data = self.data.iloc[:int(split*len(self.data))]
            else: self.data = self.data.iloc[:split]
        self.length = len(self.data)
        # load labels if we're performing fine-tuning
        self.peft = peft
        if peft:
            self.load_labels()
            if name == 'rocstories' or name == 'mctaco': c = 'correct'
            elif name == 'caters': c = 'first'
            else: c = aspect
            self.label_column = c
            if mode == 'compare':
                # we want to fine-tune on both the original and reversed comparisons
                prompts_path = f'{data_path}/{name}_prompts_compare.jsonl'
                rev_prompts_path = prompts_path.replace('.jsonl', '_reversed.jsonl')
                data = pd.read_json(prompts_path, orient='records', lines=True)
                rev_data = pd.read_json(rev_prompts_path, orient='records', lines=True)
                if split is not None:
                    if 0 <= split <= 1:
                        data = data.iloc[:int(split*len(data))]
                        rev_data = rev_data.iloc[:int(split*len(rev_data))]
                    else:
                        data = data.iloc[:split]
                        rev_data = rev_data.iloc[:split]
                self.data = pd.concat([data, rev_data]).reset_index(drop=True)
                self.length = len(self.data)

    def load_labels(self) -> None:
        labels_path = f'{data_path}/{self.name}'
        if self.name != 'rocstories' and self.mode != 'zero_shot':
            labels_path += '_pairwise_comparisons'
        labels_path += '.jsonl'
        self.labels = pd.read_json(labels_path, orient='records', lines=True)

    def get_user_prompt(self, idx: int) -> str:
        if self.name in self.grounding_datasets: return self.data.at[idx, 'prompt']
        else: return self.data.at[idx, self.aspect]
    
    def get_assistant_prompt(self) -> str:
        if self.name not in self.grounding_datasets: item = self.items[self.name]
        if self.mode == 'zero_shot':
            assert self.name in self.scoring_datasets
            content = f'I would rate the {self.aspect} of this {item} as a '
            return content
        elif self.name == 'caters':
            content = 'Between statement 1 and statement 2, the statement which appears before the other is statement '
        elif self.name == 'mctaco':
            content = 'Between choice 1 and choice 2, the more sensible option is choice '
        else:
            aspect = self.aspects_noun2adj[self.aspect]
            content = f'Between {item} 1 and {item} 2, the more {aspect} choice is {item} '
        if self.mode == 'contrast':
            content += self.choice
        return content
        
    def get_prompt(self, idx: int) -> List[Dict[str, str]]:
        user_prompt = {
            'role': 'user',
            'content': self.get_user_prompt(idx)
        }
        assistant_prompt = {
            'role': 'assistant',
            'content': self.get_assistant_prompt()
        }
        if self.peft:
            gt = str(int(self.labels.at[idx, self.label_column]))
            assistant_prompt['content'] += gt
        prompt = [user_prompt, assistant_prompt]
        return prompt
    
    def get_zero_shot_options(self) -> List[str]:
        out = None
        if self.mode == 'zero_shot' and self.name in self.scoring_datasets: out = [str(i) for i in range(1, 6)]
        return out             
    
    def get_compare_options(self) -> List[str]:
        out = None if self.mode not in ['compare', 'contrast'] else ['1', '2']
        return out    


class PPairSPEFTDataset(Dataset):

    def __init__(
            self,
            dataset: PPairSDataset,
            tokenizer: AutoTokenizer,
            max_length: int=4096
    ) -> None:
        # set pad token (eos token default)
        tokenizer.pad_token = tokenizer.eos_token
        self.examples = []
        for idx in trange(dataset.length, desc='preparing data'):
            messages = dataset.get_prompt(idx)
            # apply chat template
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # if necessary, allow for continuation instead of QA
            prompt = self.check_continue(messages, prompt)
            # tokenize
            tks = tokenizer(
                prompt,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                add_special_tokens=False
            )
            self.examples.append({
                'input_ids': tks.input_ids[0],
                'attention_mask': tks.attention_mask[0],
                'labels': tks.input_ids[0]
            })

    def __len__(self) -> int: return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return self.examples[idx]  
    
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
        if space: prompt = prompt + ' '
        return prompt