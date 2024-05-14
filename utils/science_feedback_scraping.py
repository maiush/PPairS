import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm, trange


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
for i in trange(len(links), desc="fetching claims/verdicts"):
    link = links[i]
    response = requests.get(link)
    try: 
        response.raise_for_status()
    except: 
        topics.append("")
        claims.append("")
        verdicts.append("")
        continue

    page = BeautifulSoup(response.text, "html.parser")  
    sections = page.find_all("div", class_="sfc-review-reviewed-content")
    section_claims, section_verdicts = [], []
    for section in sections:
        try:
            topic = page.find("span", class_="hero__term").text

            verdict = section.find_all("div", class_="reviewed-content__row")[0].text.strip()
            assert verdict.startswith("Verdict:\n")
            verdict = verdict.split()[-1]; section_verdicts.append(verdict)

            claim = section.find_all("div", class_="reviewed-content__row")[1].text.strip()
            assert claim.startswith("Claim:\n")
            claim = [x for x in claim.split("\n")[1:] if x != ""][0]; section_claims.append(claim)

            topics.append(topic)
            claims.append(claim)
            verdicts.append(verdict)
        except:
            continue

out = pd.DataFrame(columns=["topic", "claim", "verdict"])
out["topic"] = topics
out["claim"] = claims
out["verdict"] = verdicts
out = out[out["claim"] != ""]

category_map_inverted = {
    "Incorrect": ["Inaccurate", "Incorrect", "Unsupported", "reasoning", "inaccurate"],
    "Misleading": ["Misleading", "context", "Imprecise", "but..."],
    "Correct": ["accurate", "Accurate", "correct", "Correct"]
}
category_map = {}
for k, v in category_map_inverted.items():
    for tag in v:
        category_map[tag] = k

out["verdict"] = out["verdict"].apply(lambda x: category_map[x])
out.drop_duplicates(inplace=True)
out.to_json("sciencefeedback.jsonl", orient="records", lines=True)