import os

# --------- Audio Preprocessing ----------
from audio_preprocessing.audio_loader import load_audio
from audio_preprocessing.audio_converter import convert_to_wav_mono
from audio_preprocessing.audio_normalizer import normalize_audio
from audio_preprocessing.audio_splitter import split_audio
from audio_preprocessing.noise_reduction import reduce_noise

# --------- Language Detection ----------
from language_detection.whisper_lang_detector import detect_language_whisper

# --------- Speech to Text ----------
from speech_to_text.whisper_asr import transcribe_audio

# --------- Translation ----------
from translation.tf_translator import translate_text

# --------- Utils ----------
from utils.file_utils import create_dir_if_not_exists


TEMP_DIR = "temp_audio"
CHUNKS_DIR = "audio_chunks"

TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-hi"
TARGET_LANGUAGE = "hi"   # example: "en" for English


def preprocess_audio(input_audio_path: str):
    """
    Full audio pipeline:
    - preprocessing
    - language detection
    - speech-to-text
    - translation
    """

    create_dir_if_not_exists(TEMP_DIR)
    create_dir_if_not_exists(CHUNKS_DIR)

    # 1️⃣ Load audio
    audio = load_audio(input_audio_path)

    # 2️⃣ Convert to mono WAV @16kHz
    audio = convert_to_wav_mono(audio)

    # 3️⃣ Normalize
    audio = normalize_audio(audio)

    # 4️⃣ Save clean WAV
    clean_wav_path = os.path.join(TEMP_DIR, "clean.wav")
    audio.export(clean_wav_path, format="wav")

    # 5️⃣ Noise reduction
    denoised_path = os.path.join(TEMP_DIR, "denoised.wav")
    reduce_noise(clean_wav_path, denoised_path)

    # 6️⃣ Reload denoised audio
    final_audio = load_audio(denoised_path)

    # 7️⃣ Split into chunks
    chunks = split_audio(final_audio)

    chunk_metadata = []

    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(CHUNKS_DIR, f"chunk_{i}.wav")
        chunk.export(chunk_path, format="wav")

        # 8️⃣ Language detection
        lang_result = detect_language_whisper(chunk_path)
        detected_lang = lang_result["detected_language"]

        # 9️⃣ Speech-to-text
        asr_result = transcribe_audio(
            chunk_path,
            language=detected_lang
        )

        # 🔟 Translation
        translated_text = translate_text(
            text=asr_result["text"],
            src_lang=detected_lang,
            tgt_lang=TARGET_LANGUAGE,
            model_name=TRANSLATION_MODEL
        )

        # ✅ Final chunk record
        chunk_record = {
            "chunk_id": i,
            "path": chunk_path,
            "detected_language": detected_lang,
            "language_confidence": lang_result["confidence"],
            "transcript": asr_result["text"],
            "translated_text": translated_text,
            "segments": asr_result["segments"],
            "asr_language": asr_result["language"],
            "model": asr_result["model"]
        }

        chunk_metadata.append(chunk_record)

    return chunk_metadata


# -------------------- RUNNER --------------------

if __name__ == "__main__":
    audio_file = "test.aac"

    results = preprocess_audio(audio_file)

    print("\n✅ Pipeline Completed Successfully\n")

    for r in results:
        print("\n------------------------------")
        print(f"Chunk {r['chunk_id']}")
        print(f"Detected Language : {r['detected_language']}")
        print(f"Transcript        : {r['transcript']}")
        print(f"Translated ({TARGET_LANGUAGE}) : {r['translated_text']}")
        print("------------------------------\n")