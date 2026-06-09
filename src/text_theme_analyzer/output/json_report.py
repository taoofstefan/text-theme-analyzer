"""Render an Analysis to a JSON sidecar file.

Schema version 1.0 — bump on breaking changes to the dict shape.
"""

from __future__ import annotations

import json
from pathlib import Path

from text_theme_analyzer.pipeline.model import Analysis


SCHEMA_VERSION = "1.0"


def analysis_to_dict(analysis: Analysis) -> dict:
    """Convert an Analysis into a JSON-serializable dict.

    JSON object keys must be strings, so cluster-keyed dicts (sizes, keywords,
    representatives, series) use stringified cluster ids. To keep numeric ids
    available to machine consumers, every cluster block also exposes
    `cluster_ids: list[int]`.
    """
    out: dict = {
        "schema_version": SCHEMA_VERSION,
        "metadata": analysis.metadata,
        "summary": {
            "note_count": len(analysis.notes),
            "chunk_count": len(analysis.chunks),
        },
        "keyphrases": {
            "corpus_frequency": [
                {"phrase": p, "count": c}
                for p, c in analysis.keyphrase_frequency
            ],
            "per_note": {
                nid: [{"phrase": p, "score": s} for p, s in phrases]
                for nid, phrases in analysis.keywords.items()
            },
        },
        "clusters": None,
        "timeseries": None,
        "enrichment": None,
        "files": [
            {
                "id": n.id,
                "path": str(n.path),
                "title": n.title,
                "date": n.date.isoformat() if n.date else None,
                "tags": n.tags,
                "word_count": n.word_count,
            }
            for n in analysis.notes
        ],
    }
    if analysis.clusters is not None:
        cluster_ids = sorted(int(cid) for cid in analysis.clusters.cluster_keywords.keys())
        out["clusters"] = {
            "count": len(analysis.clusters.cluster_keywords),
            "outlier_count": int(analysis.clusters.outlier_count),
            "cluster_ids": cluster_ids,
            "sizes": [
                {"cluster_id": cid, "size": int(analysis.clusters.cluster_sizes.get(cid, 0))}
                for cid in cluster_ids
            ],
            "keywords": {
                str(cid): [{"word": w, "score": float(s)} for w, s in kw]
                for cid, kw in analysis.clusters.cluster_keywords.items()
            },
            "representatives": {
                str(cid): reps
                for cid, reps in analysis.clusters.cluster_representatives.items()
            },
            "umap_2d": [
                {
                    "x": float(x), "y": float(y), "note_id": nid,
                    "cluster_id": int(analysis.clusters.assignments[i])
                    if i < len(analysis.clusters.assignments) else -1,
                }
                for i, ((x, y), nid) in enumerate(
                    zip(analysis.clusters.umap_2d, analysis.chunk_note_ids)
                )
            ],
        }
    if analysis.timeseries is not None:
        ts_cids = sorted(int(cid) for cid in analysis.timeseries.series.keys())
        out["timeseries"] = {
            "bucket": analysis.timeseries.bucket,
            "cluster_ids": ts_cids,
            "series": {
                str(cid): {b.isoformat(): int(c) for b, c in s.items()}
                for cid, s in analysis.timeseries.series.items()
            },
            "spikes": [
                {
                    "cluster_id": int(s.cluster_id),
                    "bucket": s.bucket.isoformat(),
                    "count": int(s.count),
                    "rolling_mean": float(s.rolling_mean),
                    "delta": float(s.delta),
                }
                for s in analysis.timeseries.spikes
            ],
            "stale": [
                {
                    "cluster_id": int(s.cluster_id),
                    "first_seen": s.first_seen.isoformat(),
                    "last_seen": s.last_seen.isoformat(),
                    "frequency": int(s.frequency),
                    "severity": getattr(s, "severity", "medium") or "medium",
                    "quiet_streak_buckets": int(getattr(s, "quiet_streak_buckets", 0) or 0),
                }
                for s in analysis.timeseries.stale
            ],
        }
    if analysis.enrichment is not None:
        # EnrichmentResult is a Pydantic model.
        out["enrichment"] = json.loads(analysis.enrichment.model_dump_json())
    return out


def write_json(analysis: Analysis, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "themes.json"
    out_path.write_text(
        json.dumps(analysis_to_dict(analysis), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path
