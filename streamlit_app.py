import os
import tempfile
import streamlit as st

# --------- Pipeline Imports ----------
from audio_preprocessing.audio_loader import load_audio
from audio_preprocessing.audio_converter import convert_to_wav_mono
from audio_preprocessing.audio_normalizer import normalize_audio
from audio_preprocessing.audio_splitter import split_audio
from audio_preprocessing.noise_reduction import reduce_noise

from language_detection.whisper_lang_detector import detect_language_whisper
from speech_to_text.whisper_asr import transcribe_audio
from translation.tf_translator import translate_text
from speaker_diarization.diarization_engine import diarize_chunks
from conversation_structuring.conversation_builder import build_conversation
from business_intelligence.key_points_extractor import extract_business_key_points
from utils.file_utils import create_dir_if_not_exists


# ==================== CONFIG ====================

TEMP_DIR = "temp_audio"
CHUNKS_DIR = "audio_chunks"

TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"

NLLB_LANG_MAP = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Malayalam": "mal_Mlym",
    "Kannada": "kan_Knda",
}

LANG_CODE_MAP = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "kn": "Kannada",
}


# ==================== STREAMLIT UI ====================

st.set_page_config(
    page_title="AC-MTS | Auto Conversation to Meeting Summary",
    layout="wide",
)

st.title("🎧 AC-MTS")
st.caption("Auto Conversation → Multilingual → Speaker-Aware → Business Intelligence")

st.divider()

# --------- Sidebar ---------
st.sidebar.header("⚙️ Configuration")

target_language_label = st.sidebar.selectbox(
    "Final Output Language",
    list(NLLB_LANG_MAP.keys()),
    index=1
)

enable_business_insights = st.sidebar.checkbox(
    "Enable Business Insights (LLM)",
    value=True
)

# 🔐 OpenAI Key Input (SECURE)
openai_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    placeholder="sk-..."
)

if openai_key:
    st.session_state["OPENAI_API_KEY"] = openai_key
    os.environ["OPENAI_API_KEY"] = openai_key  # runtime only

st.sidebar.caption(
    "🔐 Key is used only in memory and never stored."
)

# --------- File Upload ---------
uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["aac", "wav", "mp3", "m4a"]
)

run_btn = st.button("🚀 Process Audio", type="primary")

# ==================== PIPELINE EXECUTION ====================

if run_btn and uploaded_file:

    create_dir_if_not_exists(TEMP_DIR)
    create_dir_if_not_exists(CHUNKS_DIR)

    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    progress = st.progress(0)
    status = st.empty()

    # --------- STEP 1: Audio Preprocessing ---------
    status.info("🔊 Loading & preprocessing audio...")
    audio = load_audio(audio_path)
    audio = convert_to_wav_mono(audio)
    audio = normalize_audio(audio)

    clean_wav = os.path.join(TEMP_DIR, "clean.wav")
    audio.export(clean_wav, format="wav")

    denoised = os.path.join(TEMP_DIR, "denoised.wav")
    reduce_noise(clean_wav, denoised)

    final_audio = load_audio(denoised)
    chunks = split_audio(final_audio)

    progress.progress(15)

    # --------- STEP 2: ASR + Translation ---------
    status.info("📝 Transcribing & translating...")
    chunk_metadata = []

    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(CHUNKS_DIR, f"chunk_{i}.wav")
        chunk.export(chunk_path, format="wav")

        lang_result = detect_language_whisper(chunk_path)
        detected_lang = lang_result["detected_language"]

        asr = transcribe_audio(chunk_path, language=detected_lang)
        transcript = asr["text"].strip()

        src_code = NLLB_LANG_MAP.get(LANG_CODE_MAP.get(detected_lang, "English"))
        tgt_code = NLLB_LANG_MAP[target_language_label]

        if src_code != tgt_code:
            translated = translate_text(
                transcript, src_code, tgt_code, TRANSLATION_MODEL
            )
        else:
            translated = transcript

        chunk_metadata.append({
            "chunk_id": i,
            "path": chunk_path,
            "detected_language": detected_lang,
            "transcript": transcript,
            "translated_text": translated,
            "segments": asr["segments"],
        })

    progress.progress(40)

    # --------- STEP 3: Speaker Diarization ---------
    status.info("🎙️ Identifying speakers...")
    chunk_metadata = diarize_chunks(chunk_metadata)
    progress.progress(60)

    # --------- STEP 4: Conversation Structuring ---------
    status.info("🧹 Structuring conversation...")
    conversation = build_conversation(chunk_metadata)
    progress.progress(75)

    # --------- STEP 5: Business Intelligence ---------
    business_insights = None
    if enable_business_insights and openai_key:
        status.info("🧠 Extracting business insights...")
        business_insights = extract_business_key_points(
            conversation["conversation_text"]
        )

    progress.progress(100)
    status.success("✅ Processing completed")

    st.divider()

    # ==================== OUTPUT UI ====================

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧹 Structured Conversation")
        st.text_area(
            "Conversation",
            conversation["conversation_text"],
            height=400
        )

    with col2:
        st.subheader("📌 Timeline")
        for t in conversation["timeline"]:
            st.markdown(f"**{t['speaker']}**: {t['text']}")

    st.divider()

    if business_insights:
        st.subheader("🧠 Business Key Points")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### 🔑 Key Points")
            for p in business_insights.get("key_points", []):
                st.write("•", p)

            st.markdown("### ✅ Decisions")
            for d in business_insights.get("decisions", []):
                st.write("•", d)

        with c2:
            st.markdown("### 📝 Action Items")
            for a in business_insights.get("action_items", []):
                st.write("•", a)

            st.markdown("### 📊 Meeting Analysis")
            st.write("**Intent:**", business_insights.get("meeting_intent"))
            st.write("**Sentiment:**", business_insights.get("sentiment"))
    else:
        st.info("ℹ️ Enter OpenAI API key to enable business insights.")
