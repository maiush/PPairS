# NOTE: all models are instruction-tuned
models = {
    # mistral
    "mistral-large-123b": "mistralai/Mistral-Large-Instruct-2407", 
    "mistral-small-22b": "mistralai/Mistral-Small-Instruct-2409",
    "mistral-nemo-12b": "mistralai/Mistral-Nemo-Instruct-2407", 
     
    # llama
    "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",

    # gemma
    "gemma-2-27b": "google/gemma-2-27b-it",
    "gemma-2-9b": "google/gemma-2-9b-it",
    "gemma-2-2b": "google/gemma-2-2b-it",

    # qwen
    "qwen-2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    "qwen-2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen-2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
}

# NLG datasets are evaluated on specific aspects
dataset_aspects = {
    "newsroom": ["coherence", "fluency", "informativeness", "relevance"],
    "summeval": ["coherence", "consistency", "fluency", "relevance"],
    "hanna": ["coherence", "complexity", "empathy", "engagement", "relevance", "surprise"]
}