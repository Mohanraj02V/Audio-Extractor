import os
import whisper

MODEL_NAME = "small"

# 🔥 SAME cache directory as language detection
CACHE_DIR = r"D:\.cache\whisper"
os.makedirs(CACHE_DIR, exist_ok=True)

# Load Whisper model explicitly from D drive
model = whisper.load_model(
    MODEL_NAME,
    download_root=CACHE_DIR
)


def transcribe_audio(audio_path: str, language: str = None) -> dict:
    """
    Transcribe audio using Whisper ASR.
    """
    result = model.transcribe(
        audio_path,
        language=language,
        fp16=False
    )

    return {
        "text": result["text"],
        "segments": result["segments"],
        "language": result.get("language"),
        "model": MODEL_NAME
    }
