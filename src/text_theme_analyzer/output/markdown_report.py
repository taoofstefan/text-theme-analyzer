"""Render an Analysis to a Markdown report."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from text_theme_analyzer.pipeline.model import Analysis


def _format_date(d: date | None) -> str:
    return d.isoformat() if d else "—"


def _cluster_label(
    cid: int,
    keywords: list[tuple[str, float]],
    stable_names: dict[int, str] | None = None,
) -> str:
    """Best-effort human label: stable name first, then top c-TF-IDF keywords."""
    stable = (stable_names or {}).get(cid)
    if not keywords:
        base = f"Cluster {cid}"
        return f"{base} ({stable})" if stable else base
    top = [w for w, _ in keywords[:3] if w]
    if not top:
        return f"Cluster {cid}: (no keywords)"
    kw_label = f"Cluster {cid}: {', '.join(top)}"
    return f"{kw_label} ({stable})" if stable else kw_label


def render_markdown(analysis: Analysis, *, top_n_themes: int = 15) -> str:
    lines: list[str] = []
    lines.append("# Text Theme Analyzer — Report")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Notes analyzed:** {len(analysis.notes)}")
    lines.append(f"- **Total chunks:** {len(analysis.chunks)}")
    if "date_range" in analysis.metadata:
        lo, hi = analysis.metadata["date_range"]
        lines.append(f"- **Date range:** {lo} → {hi}")
    if analysis.clusters is not None:
        n_clusters = len(analysis.clusters.cluster_keywords)
        lines.append(f"- **Clusters found:** {n_clusters}")
        lines.append(f"- **Outlier chunks:** {analysis.clusters.outlier_count}")
    if "input_path" in analysis.metadata:
        lines.append(f"- **Source:** `{analysis.metadata['input_path']}`")
    lines.append("")

    # Top themes (keyphrase frequency)
    lines.append(f"## Top Themes (top {top_n_themes})")
    lines.append("")
    lines.append("Keyphrase document frequency — how many notes each phrase appears in.")
    lines.append("")
    if not analysis.keyphrase_frequency:
        lines.append("_No keyphrases extracted._")
    else:
        lines.append("| Rank | Theme | Notes |")
        lines.append("|---:|---|---:|")
        for i, (phrase, count) in enumerate(analysis.keyphrase_frequency[:top_n_themes], start=1):
            lines.append(f"| {i} | {phrase} | {count} |")
    lines.append("")

    stable_names = analysis.metadata.get("cluster_stable_names") or {}

    # Clusters
    if analysis.clusters is not None and analysis.clusters.cluster_keywords:
        lines.append("## Clusters")
        lines.append("")
        lines.append("Thematic groups found by BERTopic. Top keywords via c-TF-IDF.")
        lines.append("")
        note_by_id = {n.id: n for n in analysis.notes}
        lines.append("| Cluster | Size | Top keywords | Representative notes |")
        lines.append("|---|---:|---|---|")
        for cid in sorted(analysis.clusters.cluster_keywords):
            kws = analysis.clusters.cluster_keywords[cid]
            size = analysis.clusters.cluster_sizes.get(cid, 0)
            kw_str = ", ".join(w for w, _ in kws[:5])
            rep_ids = analysis.clusters.cluster_representatives.get(cid, [])
            rep_titles = [note_by_id[rid].title for rid in rep_ids if rid in note_by_id]
            rep_str = "; ".join(rep_titles[:2]) if rep_titles else "—"
            lines.append(f"| {_cluster_label(cid, kws, stable_names)} | {size} | {kw_str} | {rep_str} |")
        lines.append("")

    # Themes over time
    if analysis.timeseries is not None and analysis.timeseries.spikes:
        lines.append("## Recent Spikes")
        lines.append("")
        lines.append("Clusters with unusually high activity in a recent time bucket.")
        lines.append("")
        lines.append("| Bucket | Cluster | Count | Rolling mean | Δ |")
        lines.append("|---|---|---:|---:|---:|")
        for spike in analysis.timeseries.spikes[:10]:
            label = _cluster_label(
                spike.cluster_id,
                analysis.clusters.cluster_keywords.get(spike.cluster_id, []) if analysis.clusters else [],
                stable_names,
            )
            lines.append(
                f"| {spike.bucket.isoformat()} | {label} | {spike.count} | "
                f"{spike.rolling_mean:.1f} | {spike.delta:+.1f} |"
            )
        lines.append("")

    # Stale ideas
    if analysis.timeseries is not None and analysis.timeseries.stale:
        lines.append("## Stale-but-Recurring")
        lines.append("")
        lines.append(
            "Clusters that used to recur but have gone quiet in the recent window. "
            "Severity ladder: **strong** = frequent + long silence, **medium** = "
            "recurred then stopped, **weak** = low-frequency, extended silence."
        )
        lines.append("")
        lines.append("| Cluster | First seen | Last seen | Frequency | Severity | Quiet (buckets) |")
        lines.append("|---|---|---|---:|---|---:|")
        for s in analysis.timeseries.stale:
            label = _cluster_label(
                s.cluster_id,
                analysis.clusters.cluster_keywords.get(s.cluster_id, []) if analysis.clusters else [],
                stable_names,
            )
            severity = getattr(s, "severity", "medium") or "medium"
            quiet_streak = int(getattr(s, "quiet_streak_buckets", 0) or 0)
            lines.append(
                f"| {label} | {s.first_seen.isoformat()} | {s.last_seen.isoformat()} | "
                f"{s.frequency} | **{severity}** | {quiet_streak} |"
            )
        lines.append("")

    # Emotional tone over time.
    tone_data = analysis.metadata.get("tone_over_time", [])
    if tone_data:
        lines.append("## Emotional Tone Over Time")
        lines.append("")
        lines.append("Lightweight lexicon-based scoring (positive/negative valence, high/low arousal).")
        lines.append("")
        lines.append("| Month | Notes | Valence | Arousal |")
        lines.append("|---|---:|---:|---:|")
        for row in tone_data:
            lines.append(
                f"| {row['bucket']} | {row['count']} | {row['valence']:+.3f} | {row['arousal']:+.3f} |"
            )
        lines.append("")

    # LLM enrichment
    if analysis.enrichment is not None:
        e = analysis.enrichment

        if e.clusters:
            lines.append("## Cluster Narratives")
            lines.append("")
            lines.append("LLM-suggested names, summaries, and strong quotes per cluster.")
            lines.append("")
            cluster_kws = (
                analysis.clusters.cluster_keywords if analysis.clusters else {}
            )
            for ann in e.clusters:
                # Fall back to a deterministic name from the cluster's top
                # c-TF-IDF keywords if the LLM didn't supply one.
                display_name = (ann.name or "").strip() or _cluster_label(
                    ann.cluster_id, cluster_kws.get(ann.cluster_id, [])
                ).replace(f"Cluster {ann.cluster_id}: ", "")
                lines.append(f"### Cluster {ann.cluster_id}: {display_name}")
                lines.append("")
                lines.append(f"_{ann.emotional_tone}_")
                lines.append("")
                lines.append(ann.summary)
                lines.append("")
                if ann.top_quotes:
                    lines.append("**Strong quotes:**")
                    lines.append("")
                    for q in ann.top_quotes:
                        lines.append(f"> {q}")
                    lines.append("")
            qv = e.quote_validation
            if qv.dropped:
                lines.append(
                    f"_Quote validation: {qv.dropped}/{qv.requested} returned quotes "
                    f"were dropped because they didn't appear in the source notes._"
                )
                lines.append("")

        if e.tensions:
            lines.append("## Tensions")
            lines.append("")
            lines.append("Opposing pulls between clusters.")
            lines.append("")
            for t in e.tensions:
                lines.append(f"### {t.title}")
                lines.append("")
                lines.append(f"- **Pole A:** {t.pole_a}")
                lines.append(f"- **Pole B:** {t.pole_b}")
                if t.evidence:
                    lines.append("")
                    lines.append("**Evidence:**")
                    for ev in t.evidence:
                        lines.append(f"- {ev}")
                lines.append("")
                if t.note:
                    lines.append(f"_{t.note}_")
                    lines.append("")

        if e.article_candidates:
            lines.append("## Article Candidates")
            lines.append("")
            lines.append("Pull-quote-able article angles extracted from your notes.")
            lines.append("")
            for art in e.article_candidates:
                lines.append(f"### {art.title}")
                lines.append("")
                lines.append(art.angle)
                if art.supporting_cluster_ids:
                    lines.append("")
                    lines.append(
                        f"_Supporting clusters: {', '.join(f'#{c}' for c in art.supporting_cluster_ids)}_"
                    )
                lines.append("")

        if e.stale_recurring:
            lines.append("## Stale-but-Recurring Verdicts")
            lines.append("")
            lines.append("LLM verdict on each stale idea: promote, archive, or keep observing.")
            lines.append("")
            lines.append("| Theme | Verdict | Reasoning |")
            lines.append("|---|---|---|")
            for s in e.stale_recurring:
                lines.append(
                    f"| {s.theme} | **{s.verdict}** | {s.reasoning} |"
                )
            lines.append("")

    # Per-note keyphrases
    lines.append("## Per-Note Keyphrases")
    lines.append("")
    for note in analysis.notes:
        title = note.title
        d = _format_date(note.date)
        lines.append(f"### {title}  \n`{note.path.name}` · {d}")
        lines.append("")
        phrases = analysis.keywords.get(note.id, [])
        if phrases:
            top = ", ".join(f"{p} ({s:.2f})" for p, s in phrases[:10])
            lines.append(top)
        else:
            lines.append("_(no keyphrases)_")
        lines.append("")

    # Files analyzed
    lines.append("## Files Analyzed")
    lines.append("")
    lines.append("| File | Date | Words | Tags |")
    lines.append("|---|---|---:|---|")
    for note in sorted(analysis.notes, key=lambda n: n.date or date.min, reverse=True):
        tags = ", ".join(note.tags) if note.tags else ""
        lines.append(
            f"| `{note.path.name}` | {_format_date(note.date)} | {note.word_count} | {tags} |"
        )
    lines.append("")

    return "\n".join(lines)


def write_markdown(analysis: Analysis, output_dir: Path, *, top_n_themes: int = 15) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    text = render_markdown(analysis, top_n_themes=top_n_themes)
    out_path = output_dir / "themes-report.md"
    out_path.write_text(text, encoding="utf-8")
    return out_path
