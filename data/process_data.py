from dev.constants import data_path
import pandas as pd


# newsroom
df = pd.read_json(f"{data_path}/unprocessed/newsroom.json")
article_ids = [f"A{i}" for i in range(60) for _ in range(7)]
summary_ids = [f"S{i % 7}" for i in range(60*7)]
articles = df["source"].unique().tolist()
summaries, scores = [], []
for article in articles:
    article_data = df.loc[df["source"] == article, ["system_output", "scores"]]
    summaries += article_data["system_output"].tolist()
    scores += article_data["scores"].tolist()
df = pd.DataFrame()
df["article_id"] = article_ids
df["summary_id"] = summary_ids
df["article"] = [a for a in articles for _ in range(7)]
df["summary"] = summaries
df["scores"] = scores
for aspect in ["coherence", "fluency", "informativeness", "relevance"]:
    df[aspect] = df["scores"].apply(lambda x: x[aspect])

df.drop(columns=["scores"], inplace=True)
df.to_json(f"{data_path}/newsroom.jsonl", orient="records", lines=True)


# summeval
df = pd.read_json(f"{data_path}/unprocessed/summeval.jsonl", orient="records", lines=True)
article_ids = [f"A{i}" for i in range(100) for _ in range(16)]
summary_ids = [f"S{i % 16}" for i in range(100*16)]
articles = df["text"].unique().tolist()
summaries, scores = [], []
for article in articles:
    article_data = df.loc[df["text"] == article, ["decoded", "expert_annotations"]]
    summaries += article_data["decoded"].tolist()
    scores += article_data["expert_annotations"].tolist()
df = pd.DataFrame()
df["article_id"] = article_ids
df["summary_id"] = summary_ids
df["article"] = [a for a in articles for _ in range(16)]
df["summary"] = summaries
df["scores"] = scores
for aspect in ["coherence", "fluency", "consistency", "relevance"]:
    df[aspect] = df["scores"].apply(lambda x: sum([s[aspect] for s in x])/3)
df.drop(columns=["scores"], inplace=True)
df.to_json(f"{data_path}/summeval.jsonl", orient="records", lines=True)


# hanna
df = pd.read_csv(f"{data_path}/unprocessed/hanna.csv")
aspects = ["Relevance", "Coherence", "Empathy", "Surprise", "Engagement", "Complexity"]
prompt_ids = [f"P{i}" for i in range(96) for _ in range(11)]
story_ids = [f"S{i % 11}" for i in range(96*11)]
prompts = df["Prompt"].unique().tolist()
stories, scores = [], {aspect: [] for aspect in aspects}
for prompt in prompts:
    prompt_data = df.loc[df["Prompt"] == prompt, ["Story"]+aspects].groupby("Story").mean()
    stories += prompt_data.index.tolist()
    for aspect in aspects:
        scores[aspect] += prompt_data[aspect].tolist() 
df = pd.DataFrame()
df["prompt_id"] = prompt_ids
df["story_id"] = story_ids
df["prompt"] = [p for p in prompts for _ in range(11)]
df["story"] = stories
for aspect in aspects:
    df[aspect] = scores[aspect]
df.rename(columns={a: a.lower() for a in aspects}, inplace=True)
df.to_json(f"{data_path}/hanna.jsonl", orient="records", lines=True)