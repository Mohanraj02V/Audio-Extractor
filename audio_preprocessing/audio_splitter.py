from pydub import AudioSegment
from typing import List

def split_audio(
    audio: AudioSegment,
    chunk_duration_sec: int = 20
) -> List[AudioSegment]:
    """
    Split audio into fixed-length chunks.
    """
    chunks = []
    chunk_ms = chunk_duration_sec * 1000

    for start in range(0, len(audio), chunk_ms):
        end = start + chunk_ms
        chunks.append(audio[start:end])

    return chunks
