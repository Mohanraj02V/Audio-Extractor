import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


def cluster_speakers(
    embeddings: list[np.ndarray],
    similarity_threshold: float = 0.75
) -> list[int]:
    """
    Cluster speaker embeddings using cosine similarity.
    Returns cluster labels.
    """

    if len(embeddings) == 1:
        return [0]

    similarity_matrix = cosine_similarity(embeddings)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=1 - similarity_threshold
    )

    labels = clustering.fit_predict(1 - similarity_matrix)
    return labels.tolist()
