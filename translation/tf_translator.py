import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------- CACHE CONFIG --------------------

HF_CACHE_DIR = r"D:\.cache\huggingface"
os.makedirs(HF_CACHE_DIR, exist_ok=True)

MODEL_CACHE = {}
TOKENIZER_CACHE = {}


def load_model_and_tokenizer(model_name: str):
    if model_name not in MODEL_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=HF_CACHE_DIR
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=HF_CACHE_DIR
        )

        TOKENIZER_CACHE[model_name] = tokenizer
        MODEL_CACHE[model_name] = model

    return MODEL_CACHE[model_name], TOKENIZER_CACHE[model_name]


# -------------------- TRANSLATION --------------------

def translate_text(
    text: str,
    src_lang: str,
    tgt_lang: str,
    model_name: str
) -> str:
    """
    Translate text using NLLB model.
    src_lang / tgt_lang must be NLLB codes (e.g. eng_Latn, tam_Taml)
    """

    if not text.strip():
        return ""

    model, tokenizer = load_model_and_tokenizer(model_name)

    # 🔑 NLLB-specific handling
    tokenizer.src_lang = src_lang
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tgt_lang_id,
            max_length=256
        )

    return tokenizer.decode(
        generated_tokens[0],
        skip_special_tokens=True
    )
