from typing import List, Dict


def merge_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Merge adjacent chunks that:
    - have the same detected language
    - belong to the same output language
    """

    if not chunks:
        return []

    merged = []
    current = chunks[0].copy()

    for nxt in chunks[1:]:
        if (
            nxt["detected_language"] == current["detected_language"]
            and nxt["user_output_language"] == current["user_output_language"]
        ):
            # Merge text
            current["transcript"] += " " + nxt["transcript"]
            current["translated_text"] += " " + nxt["translated_text"]

            # Merge segments
            current["segments"].extend(nxt.get("segments", []))
        else:
            merged.append(current)
            current = nxt.copy()

    merged.append(current)
    return merged
