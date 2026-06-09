"""Render an Analysis to a self-contained HTML dashboard."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from text_theme_analyzer.pipeline.model import Analysis


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    # numpy scalars -> python scalars
    try:
        import numpy as np  # type: ignore
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def _build_chart_data(analysis: Analysis) -> dict:
    """Bundle the chart data into a single dict (serialized to JSON in the template)."""
    themes = [
        {"phrase": p, "count": c}
        for p, c in analysis.keyphrase_frequency[:15]
    ]

    over_time: dict = {"dates": [], "series": []}
    if analysis.timeseries is not None and analysis.clusters is not None:
        # Pick top 5 clusters by size.
        top5 = sorted(
            analysis.timeseries.series.keys(),
            key=lambda c: analysis.clusters.cluster_sizes.get(c, 0),
            reverse=True,
        )[:5]
        # Union of all bucket dates.
        all_dates = sorted({b for cid in top5 for b in analysis.timeseries.series.get(cid, {})})
        over_time["dates"] = [d.isoformat() for d in all_dates]
        for cid in top5:
            kws = [w for w, _ in analysis.clusters.cluster_keywords.get(cid, [])[:2]]
            label = f"#{cid} {', '.join(kws)}" if kws else f"#{cid}"
            counts = [analysis.timeseries.series.get(cid, {}).get(d, 0) for d in all_dates]
            over_time["series"].append({"label": label, "counts": counts})

    umap: list[dict] = []
    if analysis.clusters is not None:
        for i, (x, y) in enumerate(analysis.clusters.umap_2d):
            cid = analysis.clusters.assignments[i] if i < len(analysis.clusters.assignments) else -1
            note_id = analysis.chunk_note_ids[i] if i < len(analysis.chunk_note_ids) else None
            umap.append({"x": x, "y": y, "cluster": cid, "note_id": note_id})

    return {
        "themes": themes,
        "over_time": over_time,
        "umap": umap,
    }

def _cluster_label_for(analysis: Analysis, cid: int) -> str:
    kws = analysis.clusters.cluster_keywords.get(cid, []) if analysis.clusters else []
    if kws:
        return f"Cluster {cid}: {', '.join(w for w, _ in kws[:3])}"
    return f"Cluster {cid}"


def _fallback_cluster_name(analysis: Analysis, cid: int) -> str:
    """Build a deterministic cluster name from the top 2 c-TF-IDF keywords."""
    kws = analysis.clusters.cluster_keywords.get(cid, []) if analysis.clusters else []
    top = [w for w, _ in kws[:2] if w]
    return " / ".join(top) if top else f"Cluster {cid}"


def _verdict_class(v: str) -> str:
    return {
        "promote_to_project": "promote",
        "archive": "archive",
        "keep_observing": "keep",
    }.get(v, "")


def render_html(analysis: Analysis) -> str:
    template_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("dashboard.html.j2")

    # Build chart data + serializable enrichment.
    chart_data = _build_chart_data(analysis)

    # Verdicts lookup from enrichment.
    verdicts: dict[int, dict] = {}
    if analysis.enrichment is not None:
        for v in analysis.enrichment.stale_recurring:
            verdicts[v.cluster_id] = {
                "verdict": v.verdict,
                "reasoning": v.reasoning,
            }

    stale_mechanical: list[dict] = []
    if analysis.timeseries is not None:
        for s in analysis.timeseries.stale:
            v = verdicts.get(s.cluster_id)
            stale_mechanical.append({
                "label": _cluster_label_for(analysis, s.cluster_id),
                "first_seen": s.first_seen.isoformat(),
                "last_seen": s.last_seen.isoformat(),
                "frequency": s.frequency,
                "severity": getattr(s, "severity", "medium") or "medium",
                "quiet_streak_buckets": int(getattr(s, "quiet_streak_buckets", 0) or 0),
                "verdict": v["verdict"] if v else None,
                "verdict_class": _verdict_class(v["verdict"]) if v else "",
            })

    files = [
        {
            "title": n.title,
            "filename": n.path.name,
            "date": n.date.isoformat() if n.date else None,
            "tags": n.tags,
            "word_count": n.word_count,
        }
        for n in sorted(analysis.notes, key=lambda n: n.date or date.min, reverse=True)
    ]

    meta = dict(analysis.metadata)
    meta["run_date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    meta["note_count"] = len(analysis.notes)
    meta["chunk_count"] = len(analysis.chunks)
    if analysis.clusters is not None:
        meta["cluster_count"] = len(analysis.clusters.cluster_keywords)
        meta["outlier_count"] = analysis.clusters.outlier_count
    if analysis.timeseries is not None:
        meta["spike_count"] = len(analysis.timeseries.spikes)
        meta["stale_count"] = len(analysis.timeseries.stale)

    # Serialize enrichment for client-side use; also keep structured data on the side.
    enrichment = None
    if analysis.enrichment is not None:
        enrichment = json.loads(analysis.enrichment.model_dump_json())

    # Resolve LLM cluster annotations: fall back to a deterministic name
    # when the LLM didn't supply one.
    enriched_clusters: list[dict] = []
    if analysis.enrichment is not None:
        for ann in analysis.enrichment.clusters:
            name = (ann.name or "").strip() or _fallback_cluster_name(analysis, ann.cluster_id)
            enriched_clusters.append({
                "cluster_id": ann.cluster_id,
                "name": name,
                "summary": ann.summary,
                "top_quotes": ann.top_quotes,
                "emotional_tone": ann.emotional_tone,
            })

    over_time_block = chart_data["over_time"]
    umap_block = chart_data["umap"]
    # `tojson` wraps the dict in a JS string literal (with double quotes).
    # We want it as raw JSON inside a <script type="application/json"> tag, so
    # build the script content directly as a string.
    data_json = json.dumps(chart_data, default=_json_default)
    return template.render(
        meta=meta,
        data_json=data_json,
        files=files,
        stale_mechanical=stale_mechanical,
        enrichment=enrichment,
        enrichment_clusters=enriched_clusters,
        has_enrichment=bool(enriched_clusters),
        has_timeseries=bool(over_time_block.get("series")),
        has_umap=bool(umap_block),
    )


def write_html(analysis: Analysis, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "dashboard.html"
    out_path.write_text(render_html(analysis), encoding="utf-8")
    return out_path
