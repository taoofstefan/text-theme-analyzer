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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from text_theme_analyzer.pipeline.model import Analysis


HISTORY_DIRNAME = "run-history"
HISTORY_SCHEMA_VERSION = "1.0"


@dataclass
class RunSnapshot:
    """Compact per-run summary, used as the diff unit."""
    timestamp: str           # ISO 8601 UTC
    note_count: int
    chunk_count: int
    date_range: list[str] | None
    keyphrase_top: list[tuple[str, int]]  # (phrase, count) top 30
    cluster_sizes: dict[int, int]          # cluster_id -> size
    cluster_keywords: dict[int, list[str]] # cluster_id -> top 5 keywords
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
            cluster_names={int(k): v for k, v in d.get("cluster_names", {}).items()},
            spike_count=d.get("spike_count", 0),
            stale_count=d.get("stale_count", 0),
        )


def snapshot_from_analysis(analysis: Analysis, *, top_keyphrases: int = 30) -> RunSnapshot:
    """Build a RunSnapshot from a fresh Analysis. Cheap — just slices."""
    cluster_sizes: dict[int, int] = {}
    cluster_keywords: dict[int, list[str]] = {}
    if analysis.clusters is not None:
        cluster_sizes = dict(analysis.clusters.cluster_sizes)
        cluster_keywords = {
            cid: [w for w, _ in kws[:5]]
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
    added_clusters: list[int]          # cluster ids in `new` not in `old`
    removed_clusters: list[int]        # cluster ids in `old` not in `new`
    grown_clusters: list[tuple[int, int]]  # (cid, delta) where new > old
    shrunk_clusters: list[tuple[int, int]]
    added_keyphrases: list[tuple[str, int]]  # new top-N keyphrases
    dropped_keyphrases: list[str]
    new_spike_count: int
    new_stale_count: int
    old_timestamp: str
    new_timestamp: str


def diff_snapshots(old: RunSnapshot, new: RunSnapshot) -> Diff:
    """Compute a structural diff between two runs. Symmetric where it matters."""
    old_sizes = old.cluster_sizes
    new_sizes = new.cluster_sizes
    added = sorted(set(new_sizes) - set(old_sizes))
    removed = sorted(set(old_sizes) - set(new_sizes))
    grown = [
        (cid, new_sizes[cid] - old_sizes[cid])
        for cid in set(new_sizes) & set(old_sizes)
        if new_sizes[cid] > old_sizes[cid]
    ]
    shrunk = [
        (cid, old_sizes[cid] - new_sizes[cid])
        for cid in set(new_sizes) & set(old_sizes)
        if new_sizes[cid] < old_sizes[cid]
    ]
    old_phrases = {p for p, _ in old.keyphrase_top}
    new_phrases = {p for p, _ in new.keyphrase_top}
    added_kp = [(p, c) for p, c in new.keyphrase_top if p not in old_phrases]
    dropped_kp = [p for p, _ in old.keyphrase_top if p not in new_phrases]
    return Diff(
        added_clusters=added,
        removed_clusters=removed,
        grown_clusters=sorted(grown, key=lambda x: -x[1]),
        shrunk_clusters=sorted(shrunk, key=lambda x: -x[1]),
        added_keyphrases=added_kp,
        dropped_keyphrases=dropped_kp,
        new_spike_count=new.spike_count - old.spike_count,
        new_stale_count=new.stale_count - old.stale_count,
        old_timestamp=old.timestamp,
        new_timestamp=new.timestamp,
    )


def render_diff(diff: Diff, *, old: RunSnapshot, new: RunSnapshot) -> str:
    """Format a diff as a human-readable string (rich-printable)."""
    lines: list[str] = []
    lines.append(f"Run diff: {diff.old_timestamp}  →  {diff.new_timestamp}")
    lines.append(f"  Notes:           {old.note_count} → {new.note_count}  (Δ {new.note_count - old.note_count:+d})")
    lines.append(f"  Chunks:          {old.chunk_count} → {new.chunk_count}  (Δ {new.chunk_count - old.chunk_count:+d})")
    lines.append(f"  Clusters:        {len(old.cluster_sizes)} → {len(new.cluster_sizes)}")
    if diff.added_clusters:
        names = [
            f"#{cid} {new.cluster_names.get(cid) or '/'.join(new.cluster_keywords.get(cid, [])[:2])}"
            for cid in diff.added_clusters
        ]
        lines.append(f"    added:    {', '.join(names)}")
    if diff.removed_clusters:
        names = [
            f"#{cid} {old.cluster_names.get(cid) or '/'.join(old.cluster_keywords.get(cid, [])[:2])}"
            for cid in diff.removed_clusters
        ]
        lines.append(f"    removed:  {', '.join(names)}")
    if diff.grown_clusters:
        for cid, delta in diff.grown_clusters:
            name = new.cluster_names.get(cid) or "/".join(new.cluster_keywords.get(cid, [])[:2])
            lines.append(f"    grew:     #{cid} {name}  +{delta}")
    if diff.shrunk_clusters:
        for cid, delta in diff.shrunk_clusters:
            name = new.cluster_names.get(cid) or "/".join(new.cluster_keywords.get(cid, [])[:2])
            lines.append(f"    shrank:   #{cid} {name}  -{delta}")
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
