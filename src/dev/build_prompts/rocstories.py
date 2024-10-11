from PPairS.constants import data_path
from PPairS.prompts import rocstories_instruction
import pandas as pd


data = pd.read_json(f'{data_path}/rocstories.jsonl', orient='records', lines=True)
def compare_prompt(row: pd.Series) -> str:
    story = row['story']
    s1, s2 = row['statement1'], row['statement2']
    prompt = rocstories_instruction.format(
        STORY=story,
        STATEMENT1=s1,
        STATEMENT2=s2
    )
    return prompt

# column is named as such to follow aspect-convention of scoring datasets
prompts = pd.DataFrame(columns=['consistency'])
prompts['consistency'] = data.apply(compare_prompt, axis=1)
prompts.to_json(f"{data_path}/rocstories_prompts_compare.jsonl", orient="records", lines=True)