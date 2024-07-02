import os, dill, re
from dev.constants import data_storage
from pypdf import PdfReader
import nltk
from nltk import sent_tokenize
import pandas as pd
from timeout_decorator import timeout, TimeoutError
from tqdm import tqdm, trange


def process_page(page: str) -> str:
    # convert all whitespace to single spaces
    page = " ".join(page.split())
    # remove whitespace around parentheses
    page = re.sub(r'(\s([?,.!"]))|(?<=\[|\()(.*?)(?=\)|\])', lambda x: x.group().strip(), page)
    return page

TEST_SET_ACRONYMS = {
    "GHG": "greenhouse gas",
    "GHGs": "greenhouse gases",
    "CIDs": "climactic impact drivers",
    "OHC": "ocean heat content",
    "GMSL": "global mean sea level",
    "MASL": "meters above sea level",
    "CMIP5": "Coupled Model Intercomparison Project 5",
    "CMIP6": "Coupled Model Intercomparison Project 6",
    "ECS": "equilibrium climate sensitivity",
    "TCR": "transient climate response",
    "SSP": "shared socioeconomic pathway",
    "AR5": "the 5th Assessment Report",
    "ERF": "effective radiative forcing",
    "MPWP": "mid-Pliocene warm period, 3.3 to 3.0 million years ago",
    "EECO": "early Eocene climatic optimum, 50 million years ago",
    "SH": "Southern Hemisphere",
    "NH": "Northern Hemisphere",
    "SROCC": "Special Report on the Ocean and Cryosphere in a Changing Climate",
    "GMST": "global-scale annual mean surface temperature",
    "SST": "sea surface temperatures",
    "NAO/NAM": "North Atlantic Oscillation and Northern Annular Mode",
    "CDR": "carbon dioxide removal",
    "TCRE": "transient climate response to cumulative emissions of carbon dioxide",
    "PM": "particulate matter",
    "SLCFs": "short-lived climate forcers",
    "ERFaci": "effective radiative forcing from cloud–aerosol interactions",
    "INPs": "ice nucleating particles",
    "SRM": "solar radiation management",
    "P–E": "precipitation minus evaporation",
    "HC": "Hadley Circulation",
    "RCP": "representative concentration pathway",
    "RCPs": "representative concentration pathways",
    "WAIS": "West Antarctic Ice Sheet",
    "SR1.5": "Special Report on the impacts of global warming of 1.5 °C above pre-industrial levels",
    "SLE": "sea level equivalent",
    "MICI": "Marine Ice Cliff Instability",
    "ENSO": "El Niño Southern Oscillation",
    "SAH": "Sahara",
    "NEAF": "North Eastern Africa",
    "SEAF": "South Eastern Africa",
    "ESAF": "East Southern Africa",
    "MDG": "Madagascar",
    "SAS": "South Asia",
    "NWS": "North-Western South America",
    "MED": "South Europe and the Mediterranean",
    "SLR": "sea level rise",
    "EBUS": "Eastern Boundary Upwelling Systems",
    "MHWs": "marine heatwaves",
    "HABs": "harmful algal blooms",
    "GWL": "global warming levels",
    "NbS": "nature-based solutions",
    "TCs": "tropical cyclones",
    "RKRs": "representative key risks",
    "CRD": "climate resilient development",
    "NDCs": "nationally determined contributions",
    "COP26": "26th Conference of Parties",
    "DAC": "direct air capture",
    "DACCS": "direct air capture with carbon storage",
    "BECCS": "biomass energy with carbon capture and storage",
    "IPRs": "intellectual property rights",
    "SDGs": "sustainable development goals",
    "AFOLU": "agriculture, forestry, and other land use",
    "GWP100": "global warming potential over the next 100 years",
    "PES": "payment for ecosystem services",
    "BEV": "battery electric vehicles",
    "CCS": "carbon capture and storage",
    "REDD+": "reducing emissions from deforestation and forest degradation in developing countries"
}

TEST_SET_ABBREV = {
    "RCP": "representative concentration pathway",
    "SSP": "shared socioeconomic pathway"
}

def clean_text(text: str) -> str:
    # remove parenthetical references with unspecified number of references like (Author and Author, YYYY; Author et al., YYYY)
    text = re.sub(r'\((?:[A-Za-z]+(?: and [A-Za-z]+)?(?:,? \d{4})+(?:; )?|[A-Za-z]+ et al\.(?:,? \d{4})+(?:; )?)+\)', '', text)
    # remove single author parenthetical references
    text = re.sub(r'\([A-Za-z]+, \d{4}\)', '', text)
    # remove section indicators like {16.2.3.7}
    text = re.sub(r'\{.*?\}', '', text)
    return text

def acronyms(text: str) -> str:
    replaced_acronyms = set()
    for acronym, expansion in TEST_SET_ACRONYMS.items():
        pattern = re.compile(r'\b' + re.escape(acronym) + r'\b')
        if acronym not in replaced_acronyms and expansion not in text and pattern.search(text):
            text = pattern.sub(f"{expansion} ({acronym})", text, count=1)
            replaced_acronyms.add(acronym)
    for acronym, expansion in TEST_SET_ABBREV.items():
        pattern = re.compile(r'\b' + re.escape(acronym) + r'\d')
        if acronym not in replaced_acronyms and expansion not in text and pattern.search(text):
            text = pattern.sub(f"{expansion} {acronym}", text, count=1)
            replaced_acronyms.add(acronym)
    return text

@timeout(30)
def extract_text(page):
    return page.extract_text()


path = f"{data_storage}/climatex/pdf"
files = os.listdir(path)


for file in files:
    id_root = file.replace(".pdf", "").replace("_", "")
    if os.path.exists(f"{data_storage}/climatex/claims/{id_root}_processed.jsonl"): continue

    print(file)
    reader = PdfReader(f"{path}/{file}")
    pages = []
    for page in tqdm(reader.pages, desc="reading pages"):
        try:
            text = extract_text(page)
            pages.append(text)
        except:
            continue
    pages = [process_page(page) for page in tqdm(pages, desc="preprocessing pages")]
    sentences = []
    for page in tqdm(pages, desc="extracting sentences"):
        if len(page) > 0: sentences.extend(sent_tokenize(page))

    confidence_tag = r'([A-Z]+.*)\((((very )*low|(very )*high|low|medium) confidence)\).*\.'
    claims = pd.DataFrame(columns=["context", "statement", "tag"])
    for ix in trange(len(sentences), desc="extracting claims"):
        match = re.match(confidence_tag, sentences[ix])
        if match:
            claim, _, tag = match.groups()[:3]
            context = " ".join(sentences[ix-5:ix] + [claim] + sentences[ix+1:ix+6])
            claims.loc[len(claims)] = [context, claim, tag.replace(" ", "_")]

    claims["context"] = claims["context"].apply(lambda x: acronyms(clean_text(x)))
    claims["statement"] = claims["statement"].apply(lambda x: acronyms(clean_text(x)))
    claims.drop_duplicates(inplace=True)

    ids = [f"{id_root}{i+1}" for i in range(len(claims))]
    claims["statementID"] = ids
    claims.to_json(f"{data_storage}/climatex/claims/{id_root}_processed.jsonl", orient="records", lines=True)
    print("-"*50)