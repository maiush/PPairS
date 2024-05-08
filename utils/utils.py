from prompts import *
import pandas as pd

from typing import Callable, Optional
from tqdm import tqdm, trange


def get_comparisons(
        summaries: pd.DataFrame,
        chat_template: str,
        instruction: Callable[[str, str, str, str, str], str],
        answer: Callable[[str, str], str],
        choice: Optional[int]=None
) -> pd.DataFrame:
    '''
    summaries: dataframe generated in workbook 000
    chat_template: see `chat_templates` in prompts.py
    instruction: function which takes a general instruction, article, two summaries and an aspect
                 and returns the instruction prompt.
    answer: function which takes an aspect and a choice (for contrast-pairs) 
            and returns the initial answer prompt for the assistant.
    choice: for contrast-pairs

    see workbook 001 for more details
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
                        theirs_instructions[aspect],
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