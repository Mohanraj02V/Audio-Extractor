from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np

_encoder = VoiceEncoder()


def extract_embedding(wav_path: str) -> np.ndarray:
    """
    Extract speaker embedding from an audio chunk.
    """
    wav = preprocess_wav(wav_path)
    embedding = _encoder.embed_utterance(wav)
    return embedding
