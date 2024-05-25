from dev.constants import gdrive_path
from PPairS.prompts import chat_templates, aspect_noun2adj, bm_instructions, bm_theirs_compare, bm_theirs_score, bm_mine_compare

import pandas as pd
from pathlib import Path
from typing import Callable, Optional
from tqdm import tqdm


def get_comparisons(
        summaries: pd.DataFrame,
        chat_template: str,
        instruction: Callable[[str, str, str, str, str], str],
        answer: Callable[[str, str], str],
        choice: Optional[int]=None
) -> pd.DataFrame:
    '''
    summaries: dataframe generated in process_benchmark_datasets.py
    chat_template: see `chat_templates` in prompts.py
    instruction: function which takes a general instruction, article, two summaries and an aspect
                 and returns the instruction prompt.
    answer: function which takes an aspect and a choice (for contrast-pairs) 
            and returns the initial answer prompt for the assistant.
    choice: for contrast-pairs
    '''
    
    aspects = ["coherence", "consistency", "fluency", "relevance"]
    columns = ["article_id", "summary_id", "article", "summary1", "summary2"]
    columns += aspects + [f"prompt_{aspect}" for aspect in aspects]

    comparisons = pd.DataFrame(columns=columns)
    # for each article...
    for article in tqdm(summaries.article_id.unique()):
        subset = summaries.loc[summaries.article_id == article, :]
        # for every pair of summaries...
        for summary1_id in subset["summary_id"].unique():
            for summary2_id in subset["summary_id"].unique():
                row = [article, (summary1_id, summary2_id), subset["article"].iloc[0]]
                summary1 = subset.loc[subset["summary_id"] == summary1_id, "summary"].item()
                summary2 = subset.loc[subset["summary_id"] == summary2_id, "summary"].item()
                row.append(summary1)
                row.append(summary2)
                prompts = []
                # grab the pairs of scores corresponding to each summary in the pair
                for aspect in aspects:
                    s1 = subset.loc[subset["summary_id"] == summary1_id, aspect].item()
                    s2 = subset.loc[subset["summary_id"] == summary2_id, aspect].item()
                    row.append((round(s1, 2), round(s2, 2)))
                    # construct the test prompt
                    inst = instruction(
                        bm_instructions[aspect],
                        subset["article"].iloc[0],
                        summary1,
                        summary2,
                        aspect_noun2adj[aspect]
                    )
                    ans = answer(aspect_noun2adj[aspect], choice)
                    prompt = chat_template.format(
                        INSTRUCTION=inst,
                        ANSWER=ans
                    )
                    prompts.append(prompt)
                row.extend(prompts)
                comparisons.loc[len(comparisons)] = row
    return comparisons

instruction_theirs = lambda inst, article, s1, s2, aspect: bm_theirs_compare.format(
    INSTRUCTION=inst,
    ARTICLE=article,
    SUMMARY1=s1,
    SUMMARY2=s2,
    ASPECT=aspect
)
answer_theirs = lambda _1, _2: "Answer: "

instruction_mine = lambda _, article, s1, s2, aspect: bm_mine_compare.format(
    ARTICLE=article,
    SUMMARY1=s1,
    SUMMARY2=s2,
    ASPECT=aspect
)
answer_mine = lambda aspect, choice: f"Between Choice 1 and Choice 2, the more {aspect} summary is Choice {choice}"


for dataset in ["summeval", "newsroom"]:
    for model in ["mistral", "llama2", "llama3"]:

        path = f"{gdrive_path}/benchmarks/data/{dataset}-processed.jsonl"
        summaries = pd.read_json(path, orient="records", lines=True)

        # pairwise comparisons
        outpath = f"{gdrive_path}/benchmarks/prompts/{model}"
        Path(outpath).mkdir(parents=True, exist_ok=True)

        chat_template = chat_templates[model]
        for i in range(2):
            instruction = [instruction_theirs, instruction_mine][i]
            answer = [answer_theirs, answer_mine][i]
            name = ["theirs", "mine"][i]
            filename = f"{outpath}/{dataset}_{name}"
            args = [summaries, chat_template, instruction, answer]
            if name == "mine":
                for choice in [1, 2]:
                    comparisons = get_comparisons(*args, choice)
                    comparisons.to_json(f"{filename}_{choice}.jsonl", orient="records", lines=True)
            else:
                comparisons = get_comparisons(*args)
                comparisons.to_json(f"{filename}.jsonl", orient="records", lines=True)

        # direct-scoring
        for aspect in ["coherence", "consistency", "fluency", "relevance"]:
            prompts = []
            for _, row in summaries.iterrows():
                prompt = bm_theirs_score.format(
                    INSTRUCTION=bm_instructions[aspect],
                    ARTICLE=row["article"],
                    SUMMARY=row["summary"],
                    ASPECT=aspect_noun2adj[aspect]
                )
                prompt = chat_templates[model].format(
                    INSTRUCTION=prompt,
                    ANSWER="Score: "
                )
                prompts.append(prompt)
            summaries[f"prompt_{aspect}"] = prompts
        summaries.to_json(f"{outpath}/{dataset}_score.jsonl", orient="records", lines=True)