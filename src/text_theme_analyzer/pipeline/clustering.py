"""BERTopic-based clustering with custom embeddings + UMAP 2D projection.

Pipeline:
1. UMAP 5D for clustering (n_neighbors, n_components tuned for small corpora)
2. HDBSCAN for density-based cluster assignment
3. c-TF-IDF for cluster keywords (BERTopic does this)
4. reduce_outliers to reassign ambiguous points
5. UMAP 2D for the dashboard bubble chart (separate projection)
6. (optional) Tag-weighted embedding: append a per-chunk one-hot over the
   corpus's top-N tags, scaled by `tag_weight`, before clustering. This
   nudges two notes that share tags (even if their prose is dissimilar)
   toward the same cluster. `tag_weight=0` disables it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from text_theme_analyzer.pipeline.model import ClusterResult, Note, NoteChunk


def _safe_import_bertopic():
    from bertopic import BERTopic
    return BERTopic


def _safe_import_umap():
    from umap import UMAP
    return UMAP


def _safe_import_hdbscan():
    from hdbscan import HDBSCAN
    return HDBSCAN


def build_tag_matrix(
    notes: Sequence[Note],
    chunks: Sequence[NoteChunk],
    *,
    top_n_tags: int = 20,
) -> tuple[np.ndarray, list[str]]:
    """Build an (M, top_n_tags) one-hot tag matrix for a chunk corpus,
    plus the corpus's top-N tag ordering.

    Returns a tuple `(matrix, tag_columns)`:

    - `matrix` is `np.ndarray`, shape `(M, N)`, dtype `float32`. It is
      aligned to `chunks` (one row per chunk, in order). Each row's
      parent note's tags are looked up; 1.0 in any of the top-N global
      tag positions that note carries, 0.0 elsewhere.
    - `tag_columns` is a `list[str]` of length `N` in frequency-desc
      order (ties broken by first-seen order, as `Counter.most_common`
      does). Column `j` of the matrix corresponds to
      `tag_columns[j]`. This is the hook for per-tag weight tables
      (e.g. scaling column `j` by a `tag_weights[tag_columns[j]]`
      factor): callers can index into the matrix by tag *string* via
      this list.

    Notes with no tags get an all-zero row — they contribute nothing to
    the tag component, so the embedding portion drives their cluster
    assignment. If the corpus has fewer than `top_n_tags` distinct tags,
    the matrix is narrower (one column per tag). If the corpus has zero
    tags at all, returns `np.zeros((M, 0), [])`.
    """
    # Global tag frequency, sorted desc; take top N.
    tag_counts: Counter[str] = Counter()
    for n in notes:
        tag_counts.update(n.tags)
    top_tags = [t for t, _ in tag_counts.most_common(top_n_tags)]
    if not top_tags:
        # No tags in the corpus: a (M, 0) matrix hstacks to nothing.
        return np.zeros((len(chunks), 0), dtype=np.float32), []

    tag_index = {t: i for i, t in enumerate(top_tags)}
    notes_by_id = {n.id: n for n in notes}

    M = len(chunks)
    N = len(top_tags)
    mat = np.zeros((M, N), dtype=np.float32)
    for i, c in enumerate(chunks):
        note = notes_by_id.get(c.note_id)
        if note is None:
            continue
        for t in note.tags:
            j = tag_index.get(t)
            if j is not None:
                mat[i, j] = 1.0
    return mat, top_tags


def cluster_chunks(
    chunks: Sequence,
    embeddings: np.ndarray,
    *,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    umap_n_neighbors: int = 15,
    umap_n_components: int = 5,
    n_topics: int | None = None,
    tag_weight: float = 0.0,
    top_n_tags: int = 20,
    tag_matrix: np.ndarray | None = None,
    tag_columns: list[str] | None = None,
    tag_weights: dict[str, float] | None = None,
) -> ClusterResult:
    """Cluster chunk embeddings into thematic groups.

    Returns a `ClusterResult`. The order of `chunks` must match the rows of
    `embeddings`.

    When `tag_weight > 0` and `tag_matrix` is provided (shape `(M, N)` where
    M == len(chunks)), the effective embedding used for clustering is
    `np.hstack([embeddings, tag_matrix * tag_weight])`. Per-tag weights
    (T1.1b) are applied column-wise before the global `tag_weight` scale.
    The `umap_2d` projection that drives the dashboard's bubble map always
    uses the original embeddings — concatenation would shift the 2D layout
    in hard-to-interpret ways, and the dashboard map is a navigation aid,
    not a clustering view.

    `tag_weight=0` (the default) is a no-op: the effective embedding is
    exactly `embeddings`, so existing behavior is preserved bit-for-bit.
    """
    if len(chunks) == 0 or embeddings.shape[0] == 0:
        return ClusterResult(
            assignments=[],
            cluster_sizes={},
            cluster_keywords={},
            cluster_representatives={},
            umap_2d=[],
            outlier_count=0,
        )

    # Heuristics: small corpora need smaller min_cluster_size.
    n = len(chunks)
    if min_cluster_size is None:
        min_cluster_size = max(2, min(5, n // 5))
    if min_samples is None:
        min_samples = max(1, min(2, n // 10))

    # Apply tag weighting if requested. The original `embeddings` array
    # is left untouched — the 2D projection below needs it. (See
    # docstring for the dashboard-map reasoning.)
    if tag_weight and tag_weight > 0 and tag_matrix is not None and tag_matrix.shape[0] == n:
        scaled_matrix = tag_matrix.astype(np.float32)
        if tag_columns and tag_weights:
            for j, tag in enumerate(tag_columns):
                w = tag_weights.get(tag)
                if w is not None:
                    scaled_matrix[:, j] *= float(w)
        cluster_embeddings = np.hstack([embeddings, scaled_matrix * float(tag_weight)]).astype(np.float32)
    else:
        cluster_embeddings = embeddings

    UMAP = _safe_import_umap()
    HDBSCAN = _safe_import_hdbscan()
    BERTopic = _safe_import_bertopic()
    from sklearn.feature_extraction.text import CountVectorizer

    umap5 = UMAP(
        n_neighbors=min(umap_n_neighbors, max(2, n - 1)),
        n_components=min(umap_n_components, max(1, n - 2)),
        metric="cosine",
        random_state=42,
    )
    hdbscan = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    # Aggressive English stopwords + extra "filler" words that dominate short notes,
    # plus the label words and metadata values that the CSV ingest uses
    # when building a row body ("Draft:", "Tone: practical", "Status: approve", ...).
    # Without these the cluster keywords get swamped by "draft", "low",
    # "medium", "approve" — noise, not signal.
    extra_stops = [
        # English
        "the", "is", "to", "and", "of", "it", "in", "that", "this", "with",
        "for", "on", "at", "as", "be", "by", "an", "a", "or", "not", "are",
        "from", "but", "have", "has", "had", "was", "were", "will", "would",
        "can", "could", "should", "may", "might", "do", "does", "did",
        "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "their", "our", "its", "this", "these", "those",
        "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
        "all", "any", "both", "each", "few", "more", "most", "other", "some",
        "such", "no", "nor", "too", "very", "just", "about", "above", "after",
        "again", "against", "before", "below", "between", "down", "during",
        "into", "off", "out", "over", "own", "same", "so", "than", "through",
        "under", "until", "up", "while", "yes",
        # Generic writing nouns that dominate short notes
        "note", "notes", "post", "posts", "essay", "essays", "thing", "things",
        "way", "ways", "kind", "kinds", "lot", "lots", "bit", "bits",
        # CSV body labels (see pipeline/csv_ingest.py: Theme/Source/Platform/...)
        "draft", "hook", "tone", "source", "platform", "format",
        "personal", "level", "private", "risk", "status", "comment",
        "reuse", "revision", "approve", "approved", "reject", "rejected",
        "revise", "drafted",
        # CSV metadata values that bleed through as keywords
        "low", "medium", "high", "personal", "private", "public",
        "approve", "reject", "revise", "draft",
    ]
    vectorizer = CountVectorizer(
        stop_words=extra_stops,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
    )
    topic_model = BERTopic(
        umap_model=umap5,
        hdbscan_model=hdbscan,
        vectorizer_model=vectorizer,
        calculate_probabilities=False,
        verbose=False,
    )
    docs = [c.text for c in chunks]
    topics, _ = topic_model.fit_transform(docs, embeddings=cluster_embeddings)

    # reduce_outliers reassigns ambiguous points using embeddings + c-TF-IDF.
    try:
        new_topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings", embeddings=cluster_embeddings)
        topic_model.update_topics(docs, topics=new_topics, vectorizer_model=vectorizer)
        topics = new_topics
    except Exception:
        pass

    # Cluster keywords (c-TF-IDF).
    cluster_keywords: dict[int, list[tuple[str, float]]] = {}
    for topic_id in sorted(set(topics)):
        if topic_id == -1:
            continue
        words = topic_model.get_topic(topic_id) or []
        cluster_keywords[topic_id] = [(str(w), float(s)) for w, s in words]

    # Cluster sizes.
    cluster_sizes: dict[int, int] = {}
    for t in topics:
        cluster_sizes[t] = cluster_sizes.get(t, 0) + 1

    # Representatives: for each cluster, pick the chunk(s) whose embedding is
    # closest to the cluster centroid.
    cluster_representatives: dict[int, list[str]] = {}
    if cluster_keywords:
        for cid in cluster_keywords:
            indices = [i for i, t in enumerate(topics) if t == cid]
            if not indices:
                continue
            centroid = embeddings[indices].mean(axis=0)
            dists = np.linalg.norm(embeddings[indices] - centroid, axis=1)
            order = np.argsort(dists)
            top = [indices[i] for i in order[: min(3, len(order))]]
            cluster_representatives[cid] = [chunks[i].note_id for i in top]

    # UMAP 2D for the dashboard map.
    UMAP_2d = _safe_import_umap()
    umap2d = UMAP_2d(
        n_neighbors=min(umap_n_neighbors, max(2, n - 1)),
        n_components=2,
        metric="cosine",
        random_state=42,
    )
    coords2d = umap2d.fit_transform(embeddings)
    umap_2d = [(float(x), float(y)) for x, y in coords2d]

    outlier_count = sum(1 for t in topics if t == -1)

    return ClusterResult(
        assignments=list(topics),
        cluster_sizes=cluster_sizes,
        cluster_keywords=cluster_keywords,
        cluster_representatives=cluster_representatives,
        umap_2d=umap_2d,
        outlier_count=outlier_count,
    )


def note_to_cluster(
    chunk_note_ids: list[str],
    cluster_assignments: list[int],
) -> dict[str, int]:
    """Map each note to the mode of its chunk cluster IDs (skip -1 outliers)."""
    from collections import Counter
    by_note: dict[str, list[int]] = {}
    for note_id, c in zip(chunk_note_ids, cluster_assignments, strict=False):
        if c == -1:
            continue
        by_note.setdefault(note_id, []).append(c)
    out: dict[str, int] = {}
    for note_id, cs in by_note.items():
        if not cs:
            continue
        out[note_id] = Counter(cs).most_common(1)[0][0]
    return out
