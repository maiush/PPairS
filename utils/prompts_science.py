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


template_score = """\
Consider the following claim:

Claim: {CLAIM}

Is this claim correct, incorrect, or misleading? \
Responses must be a single choice."""

template_compare = """\
Compare the factual accuracy of the following two claims:

Claim 1: {CLAIM1}
Claim 2: {CLAIM2}

Question: Which claim is more factually accurate / less ambiguous? \
Responses must be a single choice."""