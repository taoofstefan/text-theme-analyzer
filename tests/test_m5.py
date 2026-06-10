"""M5 tests: emotional tone scoring, graceful LLM degradation, full e2e."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from text_theme_analyzer.config import Config, Provider
from text_theme_analyzer.llm.base import LLMError
from text_theme_analyzer.llm.factory import build_client
from text_theme_analyzer.output.markdown_report import render_markdown
from text_theme_analyzer.pipeline.model import Analysis, Note
from text_theme_analyzer.pipeline.tone import score_text, tone_label, tone_over_time

# --- tone ---

def test_score_text_positive_text_high_valence() -> None:
    s = score_text("I love this, it's amazing and wonderful. Great work, very happy!")
    assert s.valence > 0
    assert s.counts["positive"] > s.counts["negative"]


def test_score_text_negative_text_low_valence() -> None:
    s = score_text("This is terrible. I'm exhausted, frustrated, and stuck. Fail fail fail.")
    assert s.valence < 0
    assert s.counts["negative"] > s.counts["positive"]


def test_score_text_neutral_text_near_zero() -> None:
    s = score_text("The meeting is at three.")
    assert abs(s.valence) < 0.01
    assert abs(s.arousal) < 0.01


def test_score_text_high_arousal() -> None:
    s = score_text("Huge spike! Intense and fast, urgent!")
    assert s.arousal > 0


def test_score_text_low_arousal() -> None:
    s = score_text("A quiet, calm, still, soft and patient afternoon.")
    assert s.arousal < 0


def test_tone_label_format() -> None:
    s = score_text("great, amazing")
    label = tone_label(s)
    assert "-" in label
    assert len(label.split("-")) == 2


def test_tone_over_time_buckets_by_month() -> None:
    per_note = {
        "n1": "happy great wonderful",
        "n2": "terrible hate bad",
        "n3": "neutral text",
    }
    dates = {
        "n1": date(2025, 1, 15),
        "n2": date(2025, 1, 20),
        "n3": date(2025, 2, 5),
    }
    out = tone_over_time(per_note, dates, bucket="month")
    assert len(out) == 2
    # January should have a balanced score (1 positive + 1 negative).
    jan = next(r for r in out if r["bucket"] == "2025-01-01")
    assert jan["count"] == 2
    assert abs(jan["valence"]) < 0.05
    # February should be neutral.
    feb = next(r for r in out if r["bucket"] == "2025-02-01")
    assert feb["count"] == 1


def test_tone_over_time_skips_undated() -> None:
    per_note = {"n1": "happy", "n2": "sad"}
    dates = {"n1": date(2025, 1, 1), "n2": None}
    out = tone_over_time(per_note, dates, bucket="month")
    assert len(out) == 1
    assert out[0]["count"] == 1


# --- graceful LLM degradation ---

def test_factory_no_key_raises_helpful(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the env var isn't set, the factory should raise with a clear message."""
    monkeypatch.delenv("TEXTHEME_OLLAMA_API_KEY", raising=False)
    cfg = Config(provider=Provider.OLLAMA)
    with pytest.raises(LLMError, match="TEXTHEME_OLLAMA_API_KEY"):
        build_client(cfg)


# --- markdown includes tone section ---

def test_markdown_includes_tone_section() -> None:
    n = Note(
        id="n1", path=Path("n1.md"), title="happy",
        body="I love this amazing thing", date=date(2025, 1, 1),
    )
    a = Analysis(
        notes=[n],
        chunks=[],
        chunk_note_ids=[],
        keywords={"n1": []},
        keyphrase_frequency=[],
        clusters=None,
        timeseries=None,
        metadata={
            "date_range": ["2025-01-01", "2025-01-01"],
            "tone_over_time": [
                {"bucket": "2025-01-01", "count": 1, "valence": 0.05, "arousal": 0.01},
            ],
        },
    )
    md = render_markdown(a)
    assert "## Emotional Tone Over Time" in md
    assert "Valence" in md
    assert "+0.050" in md


# --- e2e: --no-llm produces all four outputs ---

def test_cli_no_llm_produces_all_outputs(tmp_path: Path) -> None:
    """End-to-end smoke: --no-llm mode writes all four artifact types without error."""
    pytest.importorskip("yake")
    from click.testing import CliRunner

    from text_theme_analyzer.cli import main

    # Need a small notes dir.
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "a.md").write_text(
        "---\ndate: 2025-04-01\ntitle: A\ntags: [test]\n---\n\nSome text about agents and design.\n",
        encoding="utf-8",
    )
    (notes_dir / "b.md").write_text(
        "---\ndate: 2025-04-15\ntitle: B\ntags: [test]\n---\n\nMore text about agent design and workflow.\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "analyze",
            str(notes_dir),
            "-o", "markdown,html,cli,json",
            "--output-dir", str(out_dir),
            "--no-llm",
            "-q",
        ],
        catch_exceptions=False,
    )
    # Allow non-zero exit if --help is the only path; but we expect success.
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert (out_dir / "themes-report.md").exists()
    assert (out_dir / "themes.json").exists()
    assert (out_dir / "dashboard.html").exists()
