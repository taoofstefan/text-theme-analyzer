"""Render a 2-column side-by-side diff dashboard for two RunSnapshots.

Self-contained static HTML: no JS, no external CSS, no network calls.
The "2-column" concept is the matched-pairs table, which shows each
matched cluster from `old` next to its counterpart in `new` (or notes
that the pair is grown / shrank / stable). Added and removed
clusters get their own sections. Keyphrase diff appears as a
footer block.

The page is meant to be opened in any browser and read top-to-bottom.
No interaction required.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader

if TYPE_CHECKING:
    from text_theme_analyzer.output.history import Diff, RunSnapshot


def _label(snap: RunSnapshot, cid: int) -> str:
    """Cluster label: LLM name if available, else top-2 keywords, else `#cid`."""
    name = snap.cluster_names.get(cid)
    if name:
        return name
    kws = snap.cluster_keywords.get(cid, [])
    if kws:
        return "/".join(kws[:2])
    return f"#{cid}"


def _matched_pair_rows(diff: Diff, old: RunSnapshot, new: RunSnapshot) -> list[dict]:
    """Build the matched-pairs table rows.

    Each row: {similarity, old_cid, old_label, old_size, new_cid, new_label,
    new_size, delta, verdict, verdict_class}. Sorted by similarity desc
    (most-confident matches first).
    """
    rows: list[dict] = []
    for oc, nc, sim in diff.matched_pairs:
        old_size = old.cluster_sizes.get(oc, 0)
        new_size = new.cluster_sizes.get(nc, 0)
        delta = new_size - old_size
        if delta > 0:
            verdict = "grew"
            verdict_class = "verdict-grew"
        elif delta < 0:
            verdict = "shrank"
            verdict_class = "verdict-shrank"
        else:
            verdict = "stable"
            verdict_class = "verdict-stable"
        rows.append({
            "similarity": f"{sim:.2f}",
            "old_cid": oc,
            "old_label": _label(old, oc),
            "old_size": old_size,
            "new_cid": nc,
            "new_label": _label(new, nc),
            "new_size": new_size,
            "delta": f"{delta:+d}",
            "verdict": verdict,
            "verdict_class": verdict_class,
        })
    return rows


def _added_rows(diff: Diff, new: RunSnapshot) -> list[dict]:
    return [
        {
            "cid": cid,
            "label": _label(new, cid),
            "size": new.cluster_sizes.get(cid, 0),
            "fingerprint": ", ".join(new.cluster_fingerprints.get(cid, [])[:5]),
        }
        for cid in diff.added_clusters
    ]


def _removed_rows(diff: Diff, old: RunSnapshot) -> list[dict]:
    return [
        {
            "cid": cid,
            "label": _label(old, cid),
            "size": old.cluster_sizes.get(cid, 0),
            "fingerprint": ", ".join(old.cluster_fingerprints.get(cid, [])[:5]),
        }
        for cid in diff.removed_clusters
    ]


def _headline(diff: Diff) -> str:
    """One-line summary that goes in the page header."""
    parts: list[str] = []
    if diff.added_clusters:
        parts.append(f"{len(diff.added_clusters)} added")
    if diff.removed_clusters:
        parts.append(f"{len(diff.removed_clusters)} removed")
    if diff.grown_clusters:
        parts.append(f"{len(diff.grown_clusters)} grew")
    if diff.shrunk_clusters:
        parts.append(f"{len(diff.shrunk_clusters)} shrank")
    if diff.stable_clusters:
        parts.append(f"{len(diff.stable_clusters)} stable")
    if not parts:
        return "No cluster changes between runs."
    return ", ".join(parts) + "."


def render_diff_html(diff: Diff, old: RunSnapshot, new: RunSnapshot) -> str:
    """Render the diff dashboard as a self-contained HTML string."""
    template_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=True,  # .j2 files don't match the html/xml default in select_autoescape
    )
    template = env.get_template("diff_dashboard.html.j2")

    return template.render(
        old_timestamp=old.timestamp,
        new_timestamp=new.timestamp,
        old_note_count=old.note_count,
        new_note_count=new.note_count,
        old_chunk_count=old.chunk_count,
        new_chunk_count=new.chunk_count,
        old_cluster_count=len(old.cluster_sizes),
        new_cluster_count=len(new.cluster_sizes),
        headline=_headline(diff),
        matched_rows=_matched_pair_rows(diff, old, new),
        added_rows=_added_rows(diff, new),
        removed_rows=_removed_rows(diff, old),
        added_keyphrases=diff.added_keyphrases[:8],
        dropped_keyphrases=diff.dropped_keyphrases[:8],
        spike_delta=diff.new_spike_count,
        stale_delta=diff.new_stale_count,
    )
