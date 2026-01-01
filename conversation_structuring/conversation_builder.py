from typing import List, Dict
from conversation_structuring.chunk_merger import merge_chunks
from conversation_structuring.text_cleaner import clean_text


def build_conversation(chunks: list[dict]) -> dict:
    """
    Build speaker-aware conversation structure.
    """

    conversation_lines = []
    timeline = []

    prev_speaker = None
    buffer_text = []

    for idx, chunk in enumerate(chunks):
        speaker = chunk.get("speaker_id", "Speaker 1")
        text = chunk.get("translated_text", "").strip()

        if not text:
            continue

        # If same speaker, accumulate
        if speaker == prev_speaker:
            buffer_text.append(text)
        else:
            # Flush previous speaker block
            if buffer_text and prev_speaker:
                combined_text = " ".join(buffer_text)
                conversation_lines.append(f"{prev_speaker}: {combined_text}")
                timeline.append({
                    "index": len(timeline),
                    "speaker": prev_speaker,
                    "language": chunk.get("user_output_language"),
                    "text": combined_text
                })

            # Start new speaker block
            prev_speaker = speaker
            buffer_text = [text]

    # Flush last speaker block
    if buffer_text and prev_speaker:
        combined_text = " ".join(buffer_text)
        conversation_lines.append(f"{prev_speaker}: {combined_text}")
        timeline.append({
            "index": len(timeline),
            "speaker": prev_speaker,
            "language": chunks[-1].get("user_output_language"),
            "text": combined_text
        })

    conversation_text = "\n".join(conversation_lines)

    return {
        "conversation_text": conversation_text,
        "timeline": timeline
    }

