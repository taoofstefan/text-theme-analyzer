"""M4 tests: HTML dashboard renderer."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from text_theme_analyzer.output.html_dashboard import render_html, write_html
from text_theme_analyzer.pipeline.model import Analysis, ClusterResult, Note, NoteChunk


def _fake_analysis() -> Analysis:
    n1 = Note(
        id="n1", path=Path("n1.md"), title="Agent design",
        body="The agent loop is a tree. The model decides when to call tools.",
        date=date(2025, 4, 1), tags=["ai"],
    )
    n2 = Note(
        id="n2", path=Path("n2.md"), title="Grift notes",
        body="Another crypto grift. The pattern is always the same.",
        date=date(2024, 8, 21), tags=["scams"],
    )
    chunks = [
        NoteChunk(note_id="n1", chunk_index=0, text=n1.body, char_offset=0),
        NoteChunk(note_id="n2", chunk_index=0, text=n2.body, char_offset=0),
    ]
    cluster_result = ClusterResult(
        assignments=[0, 1],
        cluster_sizes={0: 1, 1: 1},
        cluster_keywords={
            0: [("agent", 0.3), ("model", 0.2), ("loop", 0.1)],
            1: [("grift", 0.4), ("pattern", 0.2), ("crypto", 0.1)],
        },
        cluster_representatives={0: ["n1"], 1: ["n2"]},
        umap_2d=[(0.0, 0.0), (1.0, 1.0)],
        outlier_count=0,
    )
    return Analysis(
        notes=[n1, n2],
        chunks=chunks,
        chunk_note_ids=["n1", "n2"],
        keywords={"n1": [("agent", 0.5)], "n2": [("grift", 0.5)]},
        keyphrase_frequency=[("agent", 1), ("grift", 1)],
        clusters=cluster_result,
        timeseries=None,
        enrichment=None,
        metadata={"date_range": ["2024-08-21", "2025-04-01"]},
    )


def test_render_html_contains_sections() -> None:
    html = render_html(_fake_analysis())
    assert "<!doctype html>" in html
    assert "id=\"summary\"" in html
    assert "id=\"top-themes\"" in html
    assert "id=\"cluster-map\"" in html
    assert "id=\"files\"" in html
    # Data injection.
    assert 'id="tta-data"' in html
    assert "agent" in html
    assert "grift" in html


def test_render_html_includes_enrichment_sections_when_present() -> None:
    from text_theme_analyzer.llm.schemas import (
        ClusterAnnotation,
        EnrichmentResult,
        Tension,
    )
    a = _fake_analysis()
    a.enrichment = EnrichmentResult(
        clusters=[
            ClusterAnnotation(
                cluster_id=0,
                name="Agent design",
                summary="Notes on the agent loop.",
                top_quotes=["The agent loop is a tree."],
                emotional_tone="curious",
            )
        ],
        tensions=[
            Tension(
                title="Loop vs. linear",
                pole_a="agent design",
                pole_b="traditional flow",
                evidence=["a note discusses agents"],
                note="an editorial observation",
            )
        ],
    )
    html = render_html(a)
    assert "id=\"narratives\"" in html
    assert "id=\"tensions\"" in html
    assert "Agent design" in html
    assert "Loop vs. linear" in html


def test_render_html_omits_enrichment_sections_when_none() -> None:
    html = render_html(_fake_analysis())
    assert "id=\"narratives\"" not in html
    assert "id=\"tensions\"" not in html


def test_write_html_creates_file(tmp_path: Path) -> None:
    out = write_html(_fake_analysis(), tmp_path)
    assert out.exists()
    assert out.suffix == ".html"
    # File should be small.
    assert out.stat().st_size < 200_000


def test_html_data_injection_is_valid_json() -> None:
    import json
    import re
    html = render_html(_fake_analysis())
    m = re.search(r'<script id="tta-data" type="application/json">(.+?)</script>', html, re.DOTALL)
    assert m is not None
    data = json.loads(m.group(1))
    assert "themes" in data
    assert "umap" in data
    assert isinstance(data["umap"], list)
