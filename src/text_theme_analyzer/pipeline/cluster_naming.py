"""Persistent cluster names across runs (T2.3).

After each run we save the centroid + best-available name for every
cluster to `{output_dir}/cluster-names.json`. On the next run, each new
cluster's centroid is compared to the saved catalog; if cosine
similarity exceeds a threshold (default 0.85), the saved name is reused.
Otherwise the cluster is treated as new.

The catalog is keyed by a deterministic hash of the centroid so
renaming a cluster in one run does not orphan its history.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from text_theme_analyzer.pipeline.model import Analysis, ClusterResult

# Cosine-similarity threshold above which a new cluster is considered
# the same theme as a previously-saved one. Tuned conservatively: small
# corpora can shift between runs, and we only want to merge when the
# overlap is obvious.
DEFAULT_NAME_MATCH_THRESHOLD = 0.85

CLUSTER_NAMES_FILENAME = "cluster-names.json"


def _hash_vector(v: np.ndarray) -> str:
    """Return a stable, short hex key for a normalized centroid vector."""
    # Normalize so tiny scale differences don't change the key; round to
    # float32 to keep the key stable across platforms/Python versions.
    norm = np.linalg.norm(v)
    if norm == 0:
        normalized = v.astype(np.float32)
    else:
        normalized = (v / norm).astype(np.float32)
    # Pack as compact bytes and take the first 16 hex chars (64 bits).
    packed = b"".join(struct.pack("f", float(x)) for x in normalized[:16])
    return packed.hex()[:16]


def _centroid_from_result(result: ClusterResult, cid: int, embeddings: np.ndarray) -> np.ndarray | None:
    """Mean embedding of all chunks assigned to `cid`."""
    indices = [i for i, c in enumerate(result.assignments) if c == cid]
    if not indices:
        return None
    return embeddings[indices].mean(axis=0)


def _build_cluster_name(analysis: Analysis, cid: int) -> str:
    """Best available human-readable name for a cluster.

    Priority:
    1. LLM enrichment name (if present for this cluster).
    2. c-TF-IDF keyword pair fallback.
    """
    enrichment = analysis.enrichment
    if enrichment is not None:
        for ann in enrichment.clusters:
            if ann.cluster_id == cid and ann.name:
                return ann.name
    keywords = []
    if analysis.clusters is not None:
        keywords = [w for w, _ in analysis.clusters.cluster_keywords.get(cid, [])[:2]]
    return " / ".join(keywords) if keywords else f"cluster {cid}"


def build_name_catalog(
    analysis: Analysis,
    embeddings: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Create a centroid-keyed catalog from the current run.

    `embeddings` must be parallel to `analysis.clusters.assignments`.
    """
    if analysis.clusters is None:
        return {}
    catalog: dict[str, dict[str, Any]] = {}
    for cid in analysis.clusters.cluster_keywords:
        centroid = _centroid_from_result(analysis.clusters, cid, embeddings)
        if centroid is None:
            continue
        key = _hash_vector(centroid)
        catalog[key] = {
            "name": _build_cluster_name(analysis, cid),
            "centroid": centroid.astype(np.float32).tolist(),
            "last_seen_cid": cid,
        }
    return catalog


def load_name_catalog(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Load the previous run's catalog, if any."""
    path = output_dir / CLUSTER_NAMES_FILENAME
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def save_name_catalog(output_dir: Path, catalog: dict[str, dict[str, Any]]) -> Path:
    """Persist the catalog for the next run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / CLUSTER_NAMES_FILENAME
    path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def resolve_stable_name(
    centroid: np.ndarray,
    catalog: dict[str, dict[str, Any]],
    *,
    threshold: float = DEFAULT_NAME_MATCH_THRESHOLD,
) -> str | None:
    """Return the saved name for `centroid` if similarity > threshold."""
    if not catalog:
        return None
    best_key: str | None = None
    best_score = -1.0
    for key, entry in catalog.items():
        saved = np.asarray(entry.get("centroid") or [], dtype=np.float32)
        if saved.shape != centroid.shape:
            continue
        score = _cosine_similarity(centroid, saved)
        if score > best_score:
            best_score = score
            best_key = key
    if best_key is not None and best_score >= threshold:
        return catalog[best_key].get("name")
    return None


def resolve_stable_names(
    analysis: Analysis,
    embeddings: np.ndarray,
    catalog: dict[str, dict[str, Any]],
    *,
    threshold: float = DEFAULT_NAME_MATCH_THRESHOLD,
) -> dict[int, str]:
    """Map each current cluster id to a stable name from `catalog`, if any.

    Returns {cid: stable_name}. Clusters without a match are omitted.
    """
    if analysis.clusters is None:
        return {}
    out: dict[int, str] = {}
    for cid in analysis.clusters.cluster_keywords:
        centroid = _centroid_from_result(analysis.clusters, cid, embeddings)
        if centroid is None:
            continue
        name = resolve_stable_name(centroid, catalog, threshold=threshold)
        if name is not None:
            out[cid] = name
    return out


def merge_catalogs(
    old_catalog: dict[str, dict[str, Any]],
    new_catalog: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge a new catalog over an old one.

    New entries win on key collision so the latest name/centroid for a
    stable theme is preserved.
    """
    merged = dict(old_catalog)
    merged.update(new_catalog)
    return merged
