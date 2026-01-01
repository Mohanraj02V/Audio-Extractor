import numpy as np
import librosa
import noisereduce as nr
import soundfile as sf

def reduce_noise(
    wav_path: str,
    output_path: str,
    sample_rate: int = 16000
) -> None:
    """
    Perform noise reduction on WAV audio.
    """
    audio, sr = librosa.load(wav_path, sr=sample_rate, mono=True)

    reduced_noise = nr.reduce_noise(
        y=audio,
        sr=sr,
        prop_decrease=0.8
    )

    sf.write(output_path, reduced_noise, sr)
