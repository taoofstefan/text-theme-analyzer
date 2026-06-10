"""Run history: persist a compact summary of each run for later diffing.

Each successful analyze run writes a single JSON file at
``{output_dir}/run-history/{ISO timestamp}.json``. The summary is intentionally
small (a few KB) and contains only what you need to spot *change over time*:
top keyphrases, cluster sizes, timeseries spikes, and enrichment cluster names.

A separate ``text-analyzer diff`` subcommand loads two such files and prints
a comparison.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from text_theme_analyzer.pipeline.model import Analysis


HISTORY_DIRNAME = "run-history"
HISTORY_SCHEMA_VERSION = "1.0"

# How many top c-TF-IDF words to keep per cluster for fingerprint matching.
# Top-8 is the same count the LLM bundle shows (enrichment.py::build_bundle);
# reusing the same count means the fingerprint is "what the LLM would see",
# which is the strongest single signal of cluster identity across runs.
FINGERPRINT_WORDS = 8

# Default cosine-similarity threshold below which two cluster fingerprints
# are considered "not the same theme". Sanity-checked on the existing
# fake-analysis fixtures: identical fingerprints -> 1.0; disjoint -> 0.0;
# 6-of-8 overlap with a shared corpus -> ~0.85 (well above threshold).
DEFAULT_MATCH_THRESHOLD = 0.3


@dataclass
class RunSnapshot:
    """Compact per-run summary, used as the diff unit."""
    timestamp: str           # ISO 8601 UTC
    note_count: int
    chunk_count: int
    date_range: list[str] | None
    keyphrase_top: list[tuple[str, int]]  # (phrase, count) top 30
    cluster_sizes: dict[int, int]          # cluster_id -> size
    cluster_keywords: dict[int, list[str]] # cluster_id -> top 5 keywords (display)
    cluster_fingerprints: dict[int, list[str]]  # cluster_id -> top 8 c-TF-IDF words
                                             # used by diff_snapshots() to match
                                             # clusters across runs by cosine
                                             # similarity (BERTopic reassigns
                                             # cluster_ids per run). Empty dict
                                             # for pre-T1.3 snapshots — diff
                                             # falls back to raw-ID matching.
    cluster_names: dict[int, str | None]   # cluster_id -> LLM name (or None)
    spike_count: int
    stale_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HISTORY_SCHEMA_VERSION,
            "timestamp": self.timestamp,
            "note_count": self.note_count,
            "chunk_count": self.chunk_count,
            "date_range": self.date_range,
            "keyphrase_top": [
                {"phrase": p, "count": c} for p, c in self.keyphrase_top
            ],
            "cluster_sizes": {str(cid): s for cid, s in self.cluster_sizes.items()},
            "cluster_keywords": {
                str(cid): kws for cid, kws in self.cluster_keywords.items()
            },
            "cluster_fingerprints": {
                str(cid): fps for cid, fps in self.cluster_fingerprints.items()
            },
            "cluster_names": {str(cid): n for cid, n in self.cluster_names.items()},
            "spike_count": self.spike_count,
            "stale_count": self.stale_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunSnapshot":
        return cls(
            timestamp=d["timestamp"],
            note_count=d["note_count"],
            chunk_count=d["chunk_count"],
            date_range=d.get("date_range"),
            keyphrase_top=[(p["phrase"], p["count"]) for p in d.get("keyphrase_top", [])],
            cluster_sizes={int(k): v for k, v in d.get("cluster_sizes", {}).items()},
            cluster_keywords={
                int(k): list(v) for k, v in d.get("cluster_keywords", {}).items()
            },
            cluster_fingerprints={
                int(k): list(v) for k, v in d.get("cluster_fingerprints", {}).items()
            },
            cluster_names={int(k): v for k, v in d.get("cluster_names", {}).items()},
            spike_count=d.get("spike_count", 0),
            stale_count=d.get("stale_count", 0),
        )


def snapshot_from_analysis(analysis: Analysis, *, top_keyphrases: int = 30) -> RunSnapshot:
    """Build a RunSnapshot from a fresh Analysis. Cheap — just slices."""
    cluster_sizes: dict[int, int] = {}
    cluster_keywords: dict[int, list[str]] = {}
    cluster_fingerprints: dict[int, list[str]] = {}
    if analysis.clusters is not None:
        cluster_sizes = dict(analysis.clusters.cluster_sizes)
        cluster_keywords = {
            cid: [w for w, _ in kws[:5]]
            for cid, kws in analysis.clusters.cluster_keywords.items()
        }
        # Top-8 c-TF-IDF words per cluster — the fingerprint. Slightly
        # longer than the display keywords (top-5) for a more distinctive
        # match. See FINGERPRINT_WORDS docstring.
        cluster_fingerprints = {
            cid: [w for w, _ in kws[:FINGERPRINT_WORDS]]
            for cid, kws in analysis.clusters.cluster_keywords.items()
        }
    cluster_names: dict[int, str | None] = {}
    if analysis.enrichment is not None:
        for ann in analysis.enrichment.clusters:
            cluster_names[ann.cluster_id] = ann.name

    date_range = analysis.metadata.get("date_range")
    # Filename-safe variant (no colons — invalid on Windows). The full ISO
    # string is still stored inside the JSON for parsers that need it.
    ts = datetime.now(timezone.utc)
    return RunSnapshot(
        timestamp=ts.strftime("%Y-%m-%dT%H-%M-%SZ"),
        note_count=len(analysis.notes),
        chunk_count=len(analysis.chunks),
        date_range=date_range,
        keyphrase_top=analysis.keyphrase_frequency[:top_keyphrases],
        cluster_sizes=cluster_sizes,
        cluster_keywords=cluster_keywords,
        cluster_fingerprints=cluster_fingerprints,
        cluster_names=cluster_names,
        spike_count=len(analysis.timeseries.spikes) if analysis.timeseries else 0,
        stale_count=len(analysis.timeseries.stale) if analysis.timeseries else 0,
    )


def write_snapshot(snapshot: RunSnapshot, output_dir: Path) -> Path:
    """Write the snapshot to ``{output_dir}/run-history/{timestamp}.json``."""
    history_dir = output_dir / HISTORY_DIRNAME
    history_dir.mkdir(parents=True, exist_ok=True)
    out_path = history_dir / f"{snapshot.timestamp}.json"
    out_path.write_text(
        json.dumps(snapshot.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path


def list_snapshots(output_dir: Path) -> list[Path]:
    """Return snapshot files in chronological order (oldest first)."""
    history_dir = output_dir / HISTORY_DIRNAME
    if not history_dir.is_dir():
        return []
    return sorted(history_dir.glob("*.json"))


def load_snapshot(path: Path) -> RunSnapshot:
    return RunSnapshot.from_dict(json.loads(path.read_text(encoding="utf-8")))


# --- diff ---

@dataclass
class Diff:
    added_clusters: list[int]          # cluster ids in `new` not matched in `old`
    removed_clusters: list[int]        # cluster ids in `old` not matched in `new`
    grown_clusters: list[tuple[int, int]]  # (cid, delta) where new > old (matched pairs)
    shrunk_clusters: list[tuple[int, int]]  # (cid, delta) where new < old (matched pairs)
    stable_clusters: list[int]         # matched pairs with size unchanged
    added_keyphrases: list[tuple[str, int]]  # new top-N keyphrases
    dropped_keyphrases: list[str]
    new_spike_count: int
    new_stale_count: int
    old_timestamp: str
    new_timestamp: str
    matched_pairs: list[tuple[int, int, float]] = field(default_factory=list)
    # (old_cid, new_cid, similarity). Empty when both runs lack fingerprints
    # (raw-ID fallback path) or when both runs have zero clusters.


def _idf_from_runs(
    old: RunSnapshot, new: RunSnapshot,
) -> dict[str, float]:
    """Compute IDF over the union of cluster fingerprints from both runs.

    IDF = log((N + 1) / (df + 1)) + 1, the smoothed variant from
    scikit-learn's TfidfVectorizer. N is the total number of
    "documents" (here, clusters) in the union, df is the number of
    clusters in which a word appears across the union. The "+1" in
    numerator and denominator is Laplace smoothing — a word that
    appears in every cluster doesn't divide by zero.

    Words that appear in only one cluster get high IDF (distinctive).
    Words that appear in every cluster get IDF ≈ 1 (not distinctive).
    For a personal-vault corpus with 10-50 clusters, this lands most
    topical words in the 1.5-3.0 range, with stopwords near 1.0.
    """
    docs: list[list[str]] = []
    for snap in (old, new):
        for fp in snap.cluster_fingerprints.values():
            if fp:
                docs.append(list(fp))
    if not docs:
        return {}
    n = len(docs)
    df: Counter[str] = Counter()
    for d in docs:
        for w in set(d):
            df[w] += 1
    return {w: math.log((n + 1) / (df_w + 1)) + 1.0 for w, df_w in df.items()}


def _fingerprint_similarity(
    fp_a: list[str], fp_b: list[str],
    idf: dict[str, float],
) -> float:
    """Cosine similarity of two IDF-weighted keyword bags.

    Builds sparse {word: idf} vectors from each fingerprint and takes
    the dot product / (norm_a * norm_b). Words that appear in only
    one of the two fingerprints contribute zero to the dot product
    (the other side has weight 0 for them), which is the correct
    behavior — we want to score overlap, not just presence in each.

    Returns 0.0 when either fingerprint is empty (no signal to
    compare). Identical fingerprints return 1.0 exactly (modulo
    floating point).
    """
    if not fp_a or not fp_b:
        return 0.0
    sa = {w: idf.get(w, 1.0) for w in fp_a}
    sb = {w: idf.get(w, 1.0) for w in fp_b}
    # Dot product over the intersection of words.
    dot = sum(sa[w] * sb[w] for w in sa.keys() & sb.keys())
    if dot == 0.0:
        return 0.0
    na = math.sqrt(sum(v * v for v in sa.values()))
    nb = math.sqrt(sum(v * v for v in sb.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _match_clusters(
    old: RunSnapshot, new: RunSnapshot,
    *,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> tuple[list[tuple[int, int, float]], dict[int, int]]:
    """Match clusters between two snapshots by fingerprint similarity.

    Returns (matched_pairs, best_match). `matched_pairs` is a list of
    (old_cid, new_cid, similarity) tuples for clusters that pass the
    threshold. `best_match` is {old_cid: (new_cid, sim)} — every old
    cluster's single best counterpart (for the symmetry check below).

    Algorithm: greedy + symmetric. For each old cluster, find its
    best match in new. Then keep only pairs where the match is
    *mutual* — new cluster N's best old counterpart is also old
    cluster O. This avoids the "one old cluster matches two new
    clusters" failure mode that pure greedy hits.

    Fallback: if either snapshot has no fingerprints (empty
    `cluster_fingerprints`), match by raw cluster_id identity. This
    is the v1.0 behavior, kept for backwards compatibility with
    pre-T1.3 snapshots.
    """
    old_fps = old.cluster_fingerprints
    new_fps = new.cluster_fingerprints
    if not old_fps or not new_fps:
        # Fallback: raw-ID matching. Only pairs where the cid exists
        # in both snapshots and has a size in both.
        common = set(old.cluster_sizes) & set(new.cluster_sizes)
        return [(c, c, 1.0) for c in sorted(common)], {
            c: (c, 1.0) for c in sorted(common)
        }

    idf = _idf_from_runs(old, new)
    old_cids = sorted(old_fps)
    new_cids = sorted(new_fps)

    # Best-match lookup tables.
    best_for_old: dict[int, tuple[int, float]] = {}
    for oc in old_cids:
        sims = [
            (nc, _fingerprint_similarity(old_fps[oc], new_fps[nc], idf))
            for nc in new_cids
        ]
        best_for_old[oc] = max(sims, key=lambda x: x[1])

    best_for_new: dict[int, tuple[int, float]] = {}
    for nc in new_cids:
        sims = [
            (oc, _fingerprint_similarity(old_fps[oc], new_fps[nc], idf))
            for oc in old_cids
        ]
        best_for_new[nc] = max(sims, key=lambda x: x[1])

    # Symmetric + above-threshold pairs.
    matched: list[tuple[int, int, float]] = []
    for oc, (nc, sim) in best_for_old.items():
        if sim < threshold:
            continue
        if best_for_new[nc][0] != oc:
            # The new cluster's best match is some other old cluster.
            # Skip — this is the "two new clusters want the same
            # old cluster" case. The other new cluster will claim it.
            continue
        matched.append((oc, nc, sim))

    matched.sort(key=lambda x: -x[2])  # highest similarity first
    return matched, best_for_old


def diff_snapshots(
    old: RunSnapshot, new: RunSnapshot,
    *,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> Diff:
    """Compute a structural diff between two runs. Symmetric where it matters.

    When both snapshots have fingerprints, clusters are matched across
    runs by IDF-weighted cosine similarity of their top-N c-TF-IDF
    keywords (see _match_clusters). Below `threshold`, the clusters
    are considered "not the same theme" and the new cluster is
    `added` (or the old one is `removed`). Otherwise the matched
    pair is `grown` (new.size > old.size), `shrank` (new.size <
    old.size), or `stable` (sizes equal).

    When either snapshot lacks fingerprints, falls back to raw
    cluster_id matching — the v1.0 behavior.
    """
    matched, _ = _match_clusters(old, new, threshold=threshold)
    matched_old = {oc for oc, _, _ in matched}
    matched_new = {nc for _, nc, _ in matched}

    added = sorted(set(new.cluster_sizes) - matched_new)
    removed = sorted(set(old.cluster_sizes) - matched_old)

    grown: list[tuple[int, int]] = []
    shrunk: list[tuple[int, int]] = []
    stable: list[int] = []
    for oc, nc, _sim in matched:
        delta = new.cluster_sizes[nc] - old.cluster_sizes[oc]
        if delta > 0:
            grown.append((nc, delta))
        elif delta < 0:
            shrunk.append((nc, -delta))
        else:
            stable.append(nc)
    grown.sort(key=lambda x: -x[1])
    shrunk.sort(key=lambda x: -x[1])

    old_phrases = {p for p, _ in old.keyphrase_top}
    new_phrases = {p for p, _ in new.keyphrase_top}
    added_kp = [(p, c) for p, c in new.keyphrase_top if p not in old_phrases]
    dropped_kp = [p for p, _ in old.keyphrase_top if p not in new_phrases]

    return Diff(
        added_clusters=added,
        removed_clusters=removed,
        grown_clusters=grown,
        shrunk_clusters=shrunk,
        stable_clusters=stable,
        added_keyphrases=added_kp,
        dropped_keyphrases=dropped_kp,
        new_spike_count=new.spike_count - old.spike_count,
        new_stale_count=new.stale_count - old.stale_count,
        old_timestamp=old.timestamp,
        new_timestamp=new.timestamp,
        matched_pairs=matched,
    )


def render_diff(diff: Diff, *, old: RunSnapshot, new: RunSnapshot) -> str:
    """Format a diff as a human-readable string (rich-printable)."""
    def _label(cid: int, snap: RunSnapshot) -> str:
        """Cluster label: LLM name if available, else top-2 keywords."""
        name = snap.cluster_names.get(cid)
        if name:
            return name
        kws = snap.cluster_keywords.get(cid, [])
        return "/".join(kws[:2]) if kws else f"#{cid}"

    lines: list[str] = []
    lines.append(f"Run diff: {diff.old_timestamp}  →  {diff.new_timestamp}")
    lines.append(f"  Notes:           {old.note_count} → {new.note_count}  (Δ {new.note_count - old.note_count:+d})")
    lines.append(f"  Chunks:          {old.chunk_count} → {new.chunk_count}  (Δ {new.chunk_count - old.chunk_count:+d})")
    lines.append(f"  Clusters:        {len(old.cluster_sizes)} → {len(new.cluster_sizes)}")
    if diff.matched_pairs:
        avg_sim = sum(s for _, _, s in diff.matched_pairs) / len(diff.matched_pairs)
        lines.append(f"  Matched:         {len(diff.matched_pairs)} (avg similarity {avg_sim:.2f})")
    if diff.added_clusters:
        names = [f"#{cid} {_label(cid, new)}" for cid in diff.added_clusters]
        lines.append(f"    added:    {', '.join(names)}")
    if diff.removed_clusters:
        names = [f"#{cid} {_label(cid, old)}" for cid in diff.removed_clusters]
        lines.append(f"    removed:  {', '.join(names)}")
    if diff.grown_clusters:
        for cid, delta in diff.grown_clusters:
            lines.append(f"    grew:     #{cid} {_label(cid, new)}  +{delta}")
    if diff.shrunk_clusters:
        for cid, delta in diff.shrunk_clusters:
            lines.append(f"    shrank:   #{cid} {_label(cid, new)}  -{delta}")
    if diff.stable_clusters:
        names = [f"#{cid} {_label(cid, new)}" for cid in diff.stable_clusters]
        lines.append(f"    stable:   {', '.join(names)}")
    if diff.added_keyphrases:
        lines.append(f"  New top keyphrases: {', '.join(p for p, _ in diff.added_keyphrases[:8])}")
    if diff.dropped_keyphrases:
        lines.append(f"  Dropped keyphrases: {', '.join(diff.dropped_keyphrases[:8])}")
    if diff.new_spike_count or diff.new_stale_count:
        lines.append(
            f"  Spikes:           {old.spike_count} → {new.spike_count}  (Δ {diff.new_spike_count:+d})\n"
            f"  Stale:            {old.stale_count} → {new.stale_count}  (Δ {diff.new_stale_count:+d})"
        )
    return "\n".join(lines)
