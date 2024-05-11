import time
import requests
from bs4 import BeautifulSoup

import pandas as pd

from tqdm import tqdm, trange


def clean_verdicts(verdicts):
    category_map_inverted = {
        "Correct": ["Accurate", "Correct", "Mostly accurate", "Mostly correct", "Partially correct"],
        "Misleading": ["Correct but...", "Imprecise", "Lacks context", "Misleading", ""],
        "Incorrect": ["Flawed reasoning", "Inaccurate", "Incorrect", "Unsupported", "-1.4Very low", "Mostly inaccurate"]
    }
    category_map = {}
    for k, v in category_map_inverted.items():
        for tag in v:
            category_map[tag] = k

    out = []
    for v in verdicts:
        if v == "":
            out.append(v)
        elif "scientific" in v:
            if "debated" in v: 
                out.append("Misleading")
                continue
            v = float(v.split()[0])
            if v > 0:
                out.append("Correct")
            elif v < 0:
                out.append("Incorrect")
            elif v == 0:
                out.append("Misleading")
        else:
            out.append(category_map[v])

    assert len(out) == len(verdicts)
    return out

url = """https://science.feedback.org/reviews/?_pagination={PAGE}"""

links = []
for page in trange(1, 114, desc="fetching links"):
    response = requests.get(url.format(PAGE=page))
    response.raise_for_status()
    page = BeautifulSoup(response.text, "html.parser")
    for claim in page.find_all("article"):
        link = claim.find("h2", class_="story__title").find("a").get("href")
        links.append(link)

topics, claims, verdicts = [], [], []
for link in tqdm(links, desc="fetching claims/verdicts"):
    response = requests.get(link)
    try: 
        response.raise_for_status()
        claim_page = BeautifulSoup(response.text, "html.parser")
        topic = claim_page.find("span", class_="hero__term").text
        claim_text = claim_page.find("blockquote", class_="reviewed-content__quote").text
        verdict = claim_page.find("div", class_="col-content").text
        topics.append(topic)
        claims.append(claim_text)
        verdicts.append(verdict)
    except: 
        topics.append("")
        claims.append("")
        verdicts.append("")

out = pd.DataFrame(columns=["link", "topic", "claim", "verdict"])
out["link"] = links
out["topic"] = topics
out["claim"] = [x.strip() for x in claims]
out["verdict"] = [x.strip() for x in clean_verdicts(verdicts)]
out = out[out["claim"] != ""]
out.to_json("sciencefeedback.jsonl", orient="records", lines=True)