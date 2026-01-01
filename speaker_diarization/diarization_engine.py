from speaker_diarization.embedding_extractor import extract_embedding
from speaker_diarization.speaker_cluster import cluster_speakers


def diarize_chunks(chunks: list[dict]) -> list[dict]:
    """
    Assign speaker IDs to each chunk.
    """

    embeddings = []
    for chunk in chunks:
        emb = extract_embedding(chunk["path"])
        embeddings.append(emb)

    labels = cluster_speakers(embeddings)

    speaker_map = {}
    speaker_counter = 1

    for i, label in enumerate(labels):
        if label not in speaker_map:
            speaker_map[label] = f"Speaker {speaker_counter}"
            speaker_counter += 1

        chunks[i]["speaker_id"] = speaker_map[label]

    return chunks
