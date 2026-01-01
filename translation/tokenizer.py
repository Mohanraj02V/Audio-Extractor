from transformers import AutoTokenizer

TOKENIZER_CACHE = {}

def load_tokenizer(model_name: str):
    if model_name not in TOKENIZER_CACHE:
        TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(model_name)
    return TOKENIZER_CACHE[model_name]
