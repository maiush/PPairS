import pandas as pd
from PPairS.constants import data_path
from PPairS.prompts import caters_instruction, mctaco_instruction


# CaTeRS
df = pd.read_json(f'{data_path}/caters_pairwise_comparisons.jsonl', orient='records', lines=True)
df['prompt'] = df.apply(
    lambda row: caters_instruction.format(
        UNORDERED=row['unordered'],
        STATEMENT1=row['statement1'],
        STATEMENT2=row['statement2']
    ), axis=1
)
df.to_json(f'{data_path}/caters_prompts_compare.jsonl', orient='records', lines=True)
df['prompt'] = df.apply(
    lambda row: caters_instruction.format(
        UNORDERED=row['unordered'],
        STATEMENT1=row['statement2'],
        STATEMENT2=row['statement1']
    ), axis=1
)
df.to_json(f'{data_path}/caters_prompts_compare_reversed.jsonl', orient='records', lines=True)


# MC-TACO
df = pd.read_json(f'{data_path}/mctaco_pairwise_comparisons.jsonl', orient='records', lines=True)
df['prompt'] = df.apply(
    lambda row: mctaco_instruction.format(
        PASSAGE=row['context'],
        QUESTION=row['question'],
        CHOICE1=row['choice1'],
        CHOICE2=row['choice2']
    ), axis=1
)
df.to_json(f'{data_path}/mctaco_prompts_compare.jsonl', orient='records', lines=True)
df['prompt'] = df.apply(
    lambda row: mctaco_instruction.format(
        PASSAGE=row['context'],
        QUESTION=row['question'],
        CHOICE1=row['choice2'],
        CHOICE2=row['choice1']
    ), axis=1
)
df.to_json(f'{data_path}/mctaco_prompts_compare_reversed.jsonl', orient='records', lines=True)