import re


def clean_text(text: str) -> str:
    """
    Clean and normalize text for conversation & summarization.
    """

    if not text:
        return ""

    text = text.strip()

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Fix spacing before punctuation
    text = re.sub(r"\s+([?.!,])", r"\1", text)

    # Remove repeated punctuation
    text = re.sub(r"([?.!,]){2,}", r"\1", text)

    return text
