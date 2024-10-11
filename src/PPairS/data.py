from typing import Optional, List, Dict
import pandas as pd

from PPairS.constants import data_path


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
    all_datasets = scoring_datasets + comparison_datasets

    def __init__(
            self,
            name: str,
            mode: str,
            aspect: Optional[str]=None,
            choice: Optional[str]=None
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

        prompts_path = f'{data_path}/{name}_prompts_'
        if mode == 'zero_shot': prompts_path += 'zero_shot'
        else: prompts_path += 'compare'
        prompts_path += '.jsonl'
        self.data = pd.read_json(prompts_path, orient='records', lines=True)
        self.length = len(self.data)

    def get_user_prompt(self, idx: int) -> str:
        return self.data.at[idx, self.aspect]
    
    def get_assistant_prompt(self) -> str:
        item = self.items[self.name]
        if self.mode == 'zero_shot':
            assert self.name in self.scoring_datasets
            content = f'I would rate the {self.aspect} of this {item} as a '
            return content
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
        prompt = [user_prompt, assistant_prompt]
        return prompt
    
    def get_zero_shot_options(self) -> List[str]:
        assert self.name in self.scoring_datasets
        out = None if self.mode != 'zero_shot' else [str(i) for i in range(1, 6)]
        return out             
    
    def get_compare_options(self) -> List[str]:
        out = None if self.mode not in ['compare', 'contrast'] else ['1', '2']
        return out