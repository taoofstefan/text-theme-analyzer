"""Rich-based terminal summary for the `cli` output format."""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from text_theme_analyzer.pipeline.model import Analysis


def render_cli(analysis: Analysis, *, top_n_themes: int = 15) -> None:
    console = Console()
    console.print()

    # Summary panel
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("Notes:", str(len(analysis.notes)))
    summary.add_row("Chunks:", str(len(analysis.chunks)))
    if "date_range" in analysis.metadata:
        lo, hi = analysis.metadata["date_range"]
        summary.add_row("Date range:", f"{lo} -> {hi}")
    if analysis.clusters is not None:
        n_clusters = len(analysis.clusters.cluster_keywords)
        summary.add_row(
            "Clusters:",
            f"{n_clusters} ({analysis.clusters.outlier_count} outliers)",
        )
    if analysis.enrichment is not None:
        qv = analysis.enrichment.quote_validation
        summary.add_row(
            "LLM enrichment:",
            (
                f"{len(analysis.enrichment.clusters)} cluster narratives, "
                f"{len(analysis.enrichment.tensions)} tensions, "
                f"{len(analysis.enrichment.article_candidates)} article candidates"
                + (
                    f"  (dropped {qv.dropped}/{qv.requested} quotes)"
                    if qv.dropped
                    else ""
                )
            ),
        )
    console.print(Panel(summary, title="Text Theme Analyzer", border_style="cyan"))

    # Top themes
    console.print(f"\n[bold]Top {top_n_themes} themes[/bold]")
    themes_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    themes_table.add_column("#", justify="right", style="dim", width=3)
    themes_table.add_column("Phrase", style="cyan")
    themes_table.add_column("Notes", justify="right", style="green")
    for i, (phrase, count) in enumerate(analysis.keyphrase_frequency[:top_n_themes], start=1):
        themes_table.add_row(str(i), phrase, str(count))
    console.print(themes_table)

    stable_names = analysis.metadata.get("cluster_stable_names") or {}

    # Clusters
    if analysis.clusters is not None and analysis.clusters.cluster_keywords:
        console.print("\n[bold]Clusters[/bold]")
        cluster_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        cluster_table.add_column("ID", justify="right", style="dim", width=4)
        cluster_table.add_column("Size", justify="right", style="green", width=5)
        cluster_table.add_column("Top keywords", style="cyan")
        cluster_table.add_column("Stable name")
        cluster_table.add_column("Representative notes")
        for cid in sorted(analysis.clusters.cluster_keywords):
            kws = analysis.clusters.cluster_keywords[cid][:4]
            kw_str = ", ".join(w for w, _ in kws)
            size = analysis.clusters.cluster_sizes.get(cid, 0)
            note_by_id = {n.id: n for n in analysis.notes}
            rep_ids = analysis.clusters.cluster_representatives.get(cid, [])
            rep_titles = [note_by_id[i].title for i in rep_ids[:2] if i in note_by_id]
            rep_str = "; ".join(rep_titles) or "—"
            stable = stable_names.get(cid, "—")
            cluster_table.add_row(str(cid), str(size), kw_str, stable, rep_str)
        console.print(cluster_table)

    # Tensions
    if analysis.enrichment is not None and analysis.enrichment.tensions:
        console.print("\n[bold]Tensions[/bold]")
        for t in analysis.enrichment.tensions:
            console.print(
                Panel(
                    f"[cyan]{t.pole_a}[/cyan]  vs.  [magenta]{t.pole_b}[/magenta]\n\n"
                    + "\n".join(f"• {e}" for e in t.evidence)
                    + (f"\n\n[italic dim]{t.note}[/italic dim]" if t.note else ""),
                    title=t.title,
                    border_style="yellow",
                )
            )

    # Article candidates
    if analysis.enrichment is not None and analysis.enrichment.article_candidates:
        console.print("\n[bold]Article candidates[/bold]")
        for art in analysis.enrichment.article_candidates:
            supporting = ", ".join(f"#{c}" for c in art.supporting_cluster_ids)
            body = f"{art.angle}\n[dim]_Supporting clusters: {supporting}_[/dim]" if supporting else art.angle
            console.print(Panel(body, title=art.title, border_style="green"))

    # Stale verdicts
    if analysis.enrichment is not None and analysis.enrichment.stale_recurring:
        console.print("\n[bold]Stale-but-recurring verdicts[/bold]")
        st_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        st_table.add_column("Theme", style="cyan")
        st_table.add_column("Verdict", style="green")
        st_table.add_column("Reasoning")
        for s in analysis.enrichment.stale_recurring:
            st_table.add_row(s.theme, s.verdict, s.reasoning)
        console.print(st_table)

    # Spikes
    if analysis.timeseries is not None and analysis.timeseries.spikes:
        console.print("\n[bold]Recent spikes[/bold]")
        for s in analysis.timeseries.spikes[:5]:
            console.print(
                f"  • {s.bucket.isoformat()} cluster {s.cluster_id} "
                f"(count {s.count}, +{s.delta:.1f})"
            )

    # Stale (mechanical)
    if analysis.timeseries is not None and analysis.timeseries.stale:
        console.print("\n[bold]Stale-but-recurring (mechanical)[/bold]")
        for s in analysis.timeseries.stale:
            console.print(
                f"  • cluster {s.cluster_id} ({s.first_seen} -> {s.last_seen}, "
                f"freq {s.frequency})"
            )

    console.print()
