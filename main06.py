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

# --------- Translation (NLLB) ----------
from translation.tf_translator import translate_text

# --------- Speaker Diarization ----------
from speaker_diarization.diarization_engine import diarize_chunks

# --------- Conversation Structuring ----------
from conversation_structuring.conversation_builder import build_conversation

# --------- Business Intelligence (LLM) ----------
from business_intelligence.key_points_extractor import extract_business_key_points

# --------- Utils ----------
from utils.file_utils import create_dir_if_not_exists



# ==================== CONFIG ====================

TEMP_DIR = "temp_audio"
CHUNKS_DIR = "audio_chunks"

# ✅ USER SELECTED FINAL OUTPUT LANGUAGE
TARGET_LANGUAGE = "hi"   # en, hi, ta, te, ml, kn

# ✅ NLLB multilingual model (ALL language pairs)
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"

# ✅ NLLB language code mapping
NLLB_LANG_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ml": "mal_Mlym",
    "kn": "kan_Knda",
}


# ==================== PIPELINE ====================

def preprocess_audio(input_audio_path: str):
    """
    Audio → Language Detection → ASR → Translation
    """

    create_dir_if_not_exists(TEMP_DIR)
    create_dir_if_not_exists(CHUNKS_DIR)

    # 1️⃣ Load audio
    audio = load_audio(input_audio_path)

    # 2️⃣ Convert to mono WAV @16kHz
    audio = convert_to_wav_mono(audio)

    # 3️⃣ Normalize loudness
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
        detected_lang = lang_result["detected_language"].lower()
        target_lang = TARGET_LANGUAGE.lower()

        # 9️⃣ Speech-to-text
        asr_result = transcribe_audio(
            chunk_path,
            language=detected_lang
        )

        transcript_text = asr_result["text"].strip()

        # 🔟 Conditional translation (NLLB)
        src_code = NLLB_LANG_MAP.get(detected_lang)
        tgt_code = NLLB_LANG_MAP.get(target_lang)

        if src_code and tgt_code and src_code != tgt_code:
            translated_text = translate_text(
                text=transcript_text,
                src_lang=src_code,
                tgt_lang=tgt_code,
                model_name=TRANSLATION_MODEL
            )
        else:
            translated_text = transcript_text

        chunk_metadata.append({
            "chunk_id": i,
            "path": chunk_path,
            "detected_language": detected_lang,
            "user_output_language": target_lang,
            "language_confidence": lang_result["confidence"],
            "transcript": transcript_text,
            "translated_text": translated_text,
            "segments": asr_result["segments"],
            "asr_language": asr_result["language"],
            "model": asr_result["model"]
        })

    return chunk_metadata


# ==================== RUNNER ====================

if __name__ == "__main__":

    audio_file = "Test 4.aac"

    # 🔹 Step 1: Audio → ASR → Translation
    chunks = preprocess_audio(audio_file)

    # 🔹 Step 2: Speaker diarization (Phase 7.1)
    chunks = diarize_chunks(chunks)

    # 🔹 Step 3: Speaker-aware conversation structuring (Phase 7.2)
    conversation = build_conversation(chunks)

    print("\n✅ PIPELINE COMPLETED SUCCESSFULLY\n")

    print("🧹 STRUCTURED CONVERSATION\n")
    print(conversation["conversation_text"])

    print("\n📌 TIMELINE\n")
    for t in conversation["timeline"]:
        print(f"[{t['index']}] ({t['speaker']}) {t['text']}")

    # 🔹 Step 4: Business Key Points (LLM – optional)
    business_insights = extract_business_key_points(
        conversation["conversation_text"]
    )

    if business_insights:
        print("\n🧠 BUSINESS KEY POINTS (LLM)\n")
        print("Key Points:")
        for p in business_insights.get("key_points", []):
            print(f"- {p}")

        print("\nDecisions:")
        for d in business_insights.get("decisions", []):
            print(f"- {d}")

        print("\nAction Items:")
        for a in business_insights.get("action_items", []):
            print(f"- {a}")

        print(f"\nMeeting Intent   : {business_insights.get('meeting_intent')}")
        print(f"Overall Sentiment: {business_insights.get('sentiment')}")
    else:
        print("\nℹ️ Business insights skipped (OPENAI_API_KEY not set)\n")
