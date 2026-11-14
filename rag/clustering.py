"""Group chunk embeddings into topic clusters with k-means.

Each cluster has a centroid, the (unit-length) average of its chunks'
embeddings. Peers are later asked for the centroids closest to a query, so
only peers with a matching topic are asked for chunks.
"""

import math

import faiss
import numpy as np

MAX_CLUSTERS = 64


def choose_k(n_chunks, requested=0):
    """Number of clusters to build for `n_chunks` chunks.

    `requested` is the configured `n_clusters`; 0 means automatic,
    round(sqrt(n_chunks)). The result is never more than `n_chunks`, and is 0
    only for an empty corpus.
    """
    if not 0 <= requested <= MAX_CLUSTERS:
        raise ValueError(f"n_clusters must be in [0, {MAX_CLUSTERS}], got {requested}")
    if n_chunks <= 0:
        return 0
    k = requested if requested > 0 else int(round(math.sqrt(n_chunks)))
    return max(1, min(k, MAX_CLUSTERS, n_chunks))


def cluster(vectors, k, seed=1234):
    """Cluster unit vectors into at most `k` groups.

    Returns `(centroids, assignments)`: float32 `[k', dim]` unit-length
    centroids and an int array giving each vector's cluster. Clusters that end
    up empty are dropped and the rest renumbered 0..k'-1 in their original
    order, so cluster ids have no gaps.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    n, dim = vectors.shape
    if not 1 <= k <= n:
        raise ValueError(f"cannot make {k} clusters from {n} vectors")

    # spherical=True keeps centroids at unit length, so a dot product with a
    # query is a cosine similarity. min_points_per_centroid only controls a
    # "too few points" warning, which is expected for small corpora.
    kmeans = faiss.Kmeans(dim, k, niter=20, seed=seed, spherical=True, min_points_per_centroid=1)
    kmeans.train(vectors)
    _, nearest = kmeans.index.search(vectors, 1)
    assigned = nearest[:, 0]

    used = np.unique(assigned)
    renumber = np.full(k, -1, dtype=np.int64)
    renumber[used] = np.arange(len(used))

    centroids = np.array(kmeans.centroids[used], dtype=np.float32)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.where(norms > 0, norms, 1.0)
    return centroids.astype(np.float32), renumber[assigned]
