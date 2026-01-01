import librosa
import numpy as np

def extract_audio_features(wav_path: str, sample_rate: int = 16000) -> dict:
    y, sr = librosa.load(wav_path, sr=sample_rate, mono=True)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)

    # Spectral features
    zcr = librosa.feature.zero_crossing_rate(y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    # Energy
    rms = librosa.feature.rms(y=y)

    # Pitch (F0)
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]

    # ✅ CORRECT silence ratio calculation
    intervals = librosa.effects.split(y, top_db=25)

    voiced_samples = sum(end - start for start, end in intervals)
    total_samples = len(y)

    silence_ratio = 1.0 - (voiced_samples / total_samples) if total_samples > 0 else 0.0

    # Tempo (speech rhythm)
    tempo = librosa.beat.tempo(y=y, sr=sr)[0]

    features = {
        "mfcc_mean": float(np.mean(mfcc)),
        "mfcc_delta_mean": float(np.mean(mfcc_delta)),
        "zcr_mean": float(np.mean(zcr)),
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
        "rms_var": float(np.var(rms)),
        "pitch_mean": float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0,
        "pitch_var": float(np.var(pitch_values)) if len(pitch_values) > 0 else 0.0,
        "silence_ratio": float(silence_ratio),
        "tempo": float(tempo),
    }

    return features
