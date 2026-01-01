from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import tensorflow as tf

MODEL_CACHE = {}
TOKENIZER_CACHE = {}


def load_model_and_tokenizer(model_name: str):
    if model_name not in MODEL_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

        TOKENIZER_CACHE[model_name] = tokenizer
        MODEL_CACHE[model_name] = model

    return MODEL_CACHE[model_name], TOKENIZER_CACHE[model_name]


def summarize_text(
    text: str,
    model_name: str = "facebook/bart-large-cnn",
    max_input_length: int = 1024,
    max_summary_length: int = 150,
    min_summary_length: int = 40
) -> str:
    """
    Abstractive summarization using TensorFlow Transformer.
    """

    if not text or len(text.strip()) == 0:
        return ""

    model, tokenizer = load_model_and_tokenizer(model_name)

    inputs = tokenizer(
        text,
        return_tensors="tf",
        truncation=True,
        padding="max_length",
        max_length=max_input_length
    )

    summary_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_summary_length,
        min_length=min_summary_length,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary.strip()
