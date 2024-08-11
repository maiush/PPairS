# NEWSROOM

newsroom_descriptions = {
    "informativeness": "Informativeness is how well a summary of an article captures the key points of the article.",
    "relevance": "The details provided in a relevant summary of an article are consistent with details in the article.",
    "fluency": "In a fluent summary of an article the individual sentences are well-written and grammatical.",
    "coherence": "In a coherent summary of an article the phrases and sentences fit together and make sense collectively."
}

newsroom_instruction_zero-shot = """\
Consider the following article and summary:
Article: {ARTICLE}
Summary: {SUMMARY}
{DESCRIPTION} Rate the {ASPECT} of this summary from 1 to 5, where 1 represents very low {ASPECT}, \
and 5 represents excellent {ASPECT}. Responses must be a single score."""

newsroom_instruction_contrast = """\
Consider the following article:
Article: {ARTICLE}
Below are two summaries of the above article:
Summary 1: {SUMMARY1}
Summary 2: {SUMMARY2}
{DESCRIPTION} Which summary is more {ASPECT}? Responses must be a single choice."""

# SummEval

summeval_descriptions = {
    "coherence": "Coherence is the collective quality of all sentences. A coherent summary of a source should be well-structured and well-organized. It should not be a heap of related information, but should build from sentence to sentence to a coherent body of information about the source.",
    "consistency": "Consistency is the factual alignment between a summary and summarized source. A coherent summary contains only statements that are entailed by the source document.",
    "fluency": "Fluency is the quality of individual sentences. A fluent summary of a source should have no formatting problems, capitalization errors or obviously ungrammatical sentences (e.g., fragments, missing components) that make the text difficult to read.",
    "relevance": "Relevance is the selection of important content from a source. A relevant summary should include only important information from the source document."
}

summeval_instruction_zero-shot = """\
Consider the following source and summary:
Source: {ARTICLE}
Summary: {SUMMARY}
{DESCRIPTION} Rate the {ASPECT} of this summary from 1 to 5, where 1 represents very low {ASPECT}, \
and 5 represents excellent {ASPECT}. Responses must be a single score."""

summeval_instruction_contrast = """\
Consider the following source:
Source: {ARTICLE}
Below are two summaries of the above source:
Summary 1: {SUMMARY1}
Summary 2: {SUMMARY2}
{DESCRIPTION} Which summary is more {ASPECT}? Responses must be a single choice."""

# HANNA

hanna_descriptions = {
    "relevance": "A relevant story matches its prompt.",
    "coherence": "A coherent story makes sense.",
    "empathy": "An empathetic story allows the reader to understand the character's emotions.",
    "surprise": "A surprising story has a surprising end.",
    "engagement": "An engaging story allows the reader to engage with it.",
    "complexity": "A complex story is elaborate."
}

hanna_instruction_zero-shot = """\
Consider the following prompt and story:
Prompt: {ARTICLE}
Story: {SUMMARY}
{DESCRIPTION} Rate the {ASPECT} of this story from 1 to 5, where 1 represents very low {ASPECT}, \
and 5 represents excellent {ASPECT}. Responses must be a single score."""

hanna_instruction_contrast = """\
Consider the following prompt:
Prompt: {ARTICLE}
Below are two stories inspired by the above prompt:
Story 1: {SUMMARY1}
Story 2: {SUMMARY2}
{DESCRIPTION} Which story is more {ASPECT}? Responses must be a single choice."""