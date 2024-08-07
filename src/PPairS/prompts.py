'''
Chat Templates:
- since we're using `transformers` for harvesting, we manually write down chat templates for supported models.
- note that inclusion of BOS tokens, so no need to do so later!
'''

chat_templates = {
    "mistral": """<s>[INST] {INSTRUCTION} [/INST]{ANSWER}""",
    "llama2": """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{INSTRUCTION} [/INST] {ANSWER}""",
    "llama3": """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{INSTRUCTION}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{ANSWER}"""
}


'''
Benchmarking:
- benchmarking is performed on two datasets from the original PairS paper: SummEval and News Room.
'''

aspect_noun2adj = {
    "coherence": "coherent",
    "consistency": "consistent",
    "fluency": "fluent",
    "relevance": "relevant"
}

bm_instructions = {
    "coherence": "Compare the coherence of the two following summaries.",
    "fluency": "Evaluate and compare the fluency of the two following summaries.",
    "relevance": """Evaluate the relevance of the following summaries.
A summary is relevant if it captures the main points from the article, without leaving out any crucial details or adding any unnecessary or inaccurate ones.
A summary is more relevant if it uses the same or similar terms and expressions as the article.
A summary is less relevant if it omits some of the key facts from the article, or if it introduces irrelevant information that is not supported by the article.""",
    "consistency": """Evaluate and compare how two summaries consistently follow the source text.
A summary is consistent with the article if it faithfully reflects the main points, facts, and tone of the article.
A summary is inconsistent if it introduces any errors, contradictions, or distortions of the original article."""
}

bm_theirs_compare = """\
{INSTRUCTION}

Source text: {ARTICLE}

Summary A: {SUMMARY1}

Summary B: {SUMMARY2}

Question: Which summary is more {ASPECT}? \
If the summary A is more {ASPECT}, please return 'A'. \
If the summary B is more {ASPECT}, please return 'B' \
You must only return the choice."""

bm_theirs_score = """\
{INSTRUCTION}

Source text: {ARTICLE}

Summary: {SUMMARY}

Please rate on a scale from 1 to 5, where 1 represents very low {ASPECT}, \
and 5 indicates excellent {ASPECT}. You must only return an int score."""

bm_mine_compare = """\
Consider the following article: {ARTICLE}
Below are two summaries of this article:
Choice 1: {SUMMARY1}
Choice 2: {SUMMARY2} 
Which summary is more {ASPECT}?"""


'''
Domain Specialisation - Fact Checking:
- performed using fact-checked claims from the website https://science.feedback.org/
'''

fc_template_score = """\
Consider the following claim:

Claim: {CLAIM}

Is this claim correct, incorrect, or misleading? \
Responses must be a single choice."""

fc_template_compare = """\
Compare the factual accuracy of the following two claims:

Claim 1: {CLAIM1}
Claim 2: {CLAIM2}

Question: Which claim is more factually accurate / less ambiguous? \
Responses must be a single choice."""


'''
IPCC-confidence classification w/ context.
'''

ipcc_template = """\
Compare the confidence of the following two statements, considering their respective contexts:

Context 1: {CONTEXT1}
Statement 1: {STATEMENT1}

Context 2: {CONTEXT2}
Statement 2: {STATEMENT2}

Question: Which statement is expressed with higher confidence? Responses must be a single choice."""