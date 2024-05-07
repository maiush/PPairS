from textwrap import dedent


aspect_noun2verb = {
    "coherence": "coherent",
    "consistency": "consistent",
    "fluency": "fluent",
    "relevance": "relevant"
}

'''
On prompt formats:

- we manually write down the prompt formats for models used, for clarity.
- note that we've included BOS tokens too, so need to do so later!
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