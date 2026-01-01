import os
import whisper

MODEL_NAME = "small"

# 🔥 Custom cache directory on D drive
CACHE_DIR = r"D:\.cache"

os.makedirs(CACHE_DIR, exist_ok=True)

# Load whisper model with explicit cache path
model = whisper.load_model(
    MODEL_NAME,
    download_root=CACHE_DIR
)


def detect_language_whisper(audio_path: str) -> dict:
    """
    Detect language from audio using Whisper.
    """
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)
    detected_lang = max(probs, key=probs.get)

    return {
        "detected_language": detected_lang,
        "confidence": probs.get(detected_lang, 0.0),
        "scores": probs
    }
