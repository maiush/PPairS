'''
To ease further experiments, we want to preprocess the SummEval and News Room datasets in roughly the same way.

- extract original articles
- extract summaries
- extract ID's for both articles and summaries
- extract average summary scores from human raters
'''

from dev.constants import data_storage
import pandas as pd


# SummEval
path = f"{data_storage}/benchmarks/data/summeval"
df = pd.read_json(f"{path}/model_annotations.aligned.paired.jsonl", orient="records", lines=True)
out = pd.DataFrame(columns=["article_id", "model_id", "article", "summary", "coherence", "consistency", "fluency", "relevance"])
for article_id in df["id"].unique():
    summaries = df.loc[df["id"] == article_id, :]
    assert summaries["model_id"].nunique() == 16
    assert summaries["text"].nunique() == 1
    for model_id in summaries["model_id"].unique():
        ratings = summaries.loc[summaries["model_id"] == model_id, "expert_annotations"].item()
        assert len(ratings) == 3
        row = [article_id, model_id, summaries["text"].iloc[0]]
        row.append(summaries.loc[summaries["model_id"] == model_id, "decoded"].item())
        for col in ["coherence", "consistency", "fluency", "relevance"]:
            scores = [r[col] for r in ratings]
            avg_score = sum(scores) / len(scores)
            row.append(avg_score)
        out.loc[len(out)] = row
out.rename(columns={"model_id": "summary_id"}, inplace=True)
out.to_json(f"{data_storage}/benchmarks/data/summeval-processed.jsonl", orient="records", lines=True)

# News Room
path = f"{data_storage}/benchmarks/data/newsroom"
df = pd.read_csv(f"{path}/newsroom-human-eval.csv")
df.drop(columns=["ArticleTitle"], inplace=True)
column_mapping = {
    "ArticleID": "article_id",
    "System": "system_id",
    "ArticleText": "article",
    "SystemSummary": "summary",
    "CoherenceRating": "coherence",
    "FluencyRating": "fluency",
    "InformativenessRating": "consistency",
    "RelevanceRating": "relevance"
}
df.rename(columns=column_mapping, inplace=True)
df["system_id"] = df.system_id.factorize()[0]
out = pd.DataFrame(columns=df.columns)
for article in df.article_id.unique():
    for system in df.system_id.unique():
        ratings = df.loc[(df.article_id == article)&(df.system_id == system), :]
        # sanity checks
        assert len(ratings) == 3
        assert ratings["article"].nunique() == 1
        assert ratings["summary"].nunique() == 1
        row = ratings.iloc[0].copy()
        for col in ["coherence", "fluency", "consistency", "relevance"]:
            row[col] = ratings[col].mean()
        out.loc[len(out)] = row
out.rename(columns={"system_id": "summary_id"}, inplace=True)
out.to_json(f"{data_storage}/benchmarks/data/newsroom-processed.jsonl", orient="records", lines=True)