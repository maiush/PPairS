# NOTE: all models are instruction-tuned
models = {
    "mistral-large-123b": "mistralai/Mistral-Large-Instruct-2407", # 123b

    "qwen-2-72b": "Qwen/Qwen2-72B-Instruct", # 72b
    "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct", # 70b

    "yi-1.5-34b": "01-ai/Yi-1.5-34B-Chat", # 34b
    "gemma-2-27b": "google/gemma-2-27b-it", # 27b

    "mistral-nemo-12b": "mistralai/Mistral-Nemo-Instruct-2407", # 12b
    "phi-3-14b": "microsoft/Phi-3-medium-128k-instruct", # 14b

    "gemma-2-9b": "google/gemma-2-9b-it", # 9b
    "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct", # 8b

    "phi-3-3.8b": "microsoft/Phi-3-mini-128k-instruct", # 3.8b
    "gemma-2-2b": "google/gemma-2-2b-it", # 2b

    "mistral-0.1-7b": "mistralai/Mistral-7B-Instruct-v0.1", # 7b,
    "llama-2-7b": "meta-llama/Llama-2-7b-chat-hf" # 7b
}


# NLG datasets are evaluated on specific aspects
dataset_aspects = {
    "newsroom": ["coherence", "fluency", "informativeness", "relevance"],
    "summeval": ["coherence", "consistency", "fluency", "relevance"],
    "hanna": ["coherence", "complexity", "empathy", "engagement", "relevance", "surprise"]
}