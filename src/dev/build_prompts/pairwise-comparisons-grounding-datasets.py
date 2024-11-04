import random; random.seed(123456)
import pandas as pd
from PPairS.constants import data_path


# caters
caters = pd.read_json(
    f'{data_path}/caters.jsonl', orient='records', lines=True
)
caters = caters[caters['ordered'].apply(len) >= 2]
pc = pd.DataFrame(columns=['unordered', 'statement1', 'statement2', 'first'])
for _, row in caters.iterrows():
    unordered = row['unordered']
    ordered = row['ordered']
    test = random.sample(ordered, 2)
    answer = int(ordered.index(test[0]) < ordered.index(test[1])) + 1
    pc.loc[len(pc)] = [unordered] + test + [answer]
pc.to_json(f'{data_path}/caters_pairwise_comparisons.jsonl', orient='records', lines=True)


# mctaco
mctaco = pd.read_json(
    f'{data_path}/mctaco.jsonl', orient='records', lines=True
)
def can_contrast(choices: list) -> bool:
    '''
    we need at least one correct and one incorrect answer for a pairwise comparison
    '''
    return (1 in choices) & (0 in choices)
mctaco = mctaco[mctaco['correct'].apply(can_contrast)]
pc = pd.DataFrame(columns=['context', 'question', 'choice1', 'choice2', 'correct', 'type'])
for _, row in mctaco.iterrows():
    zeros, ones = [], []
    for idx, corr in enumerate(row['correct']):
        if corr == 0: zeros.append(idx)
        elif corr == 1: ones.append(idx)
    ix1, ix0 = random.sample(ones, 1)[0], random.sample(zeros, 1)[0]
    if random.uniform(0, 1) > 0.5: 
        ix1, ix0 = ix0, ix1
        correct = 2
    else: correct = 1
    pc.loc[len(pc)] = [
        row['context'],
        row['question'],
        row['choices'][ix1],
        row['choices'][ix0],
        correct,
        row['type']
    ]
pc.to_json(f'{data_path}/mctaco_pairwise_comparisons.jsonl', orient='records', lines=True)