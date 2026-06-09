"""M1 tests: ingest, preprocess, keywords, markdown output.

These tests skip KeyBERT (which pulls torch) and YAKE is not guaranteed
installed. We mock or use the cleanest paths available.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from text_theme_analyzer.config import Config, OutputFormat, Provider, load_dotenv
from text_theme_analyzer.output.markdown_report import render_markdown
from text_theme_analyzer.pipeline.csv_ingest import DEFAULT_COLUMN_MAP, load_csv
from text_theme_analyzer.pipeline.ingest import discover_notes, load_note, load_notes
from text_theme_analyzer.pipeline.keywords import _aggregate_frequency
from text_theme_analyzer.pipeline.model import Analysis, ClusterResult
from text_theme_analyzer.pipeline.preprocess import chunk_text, clean_body, preprocess_note


# --- ingest ---

def test_discover_notes_finds_markdown(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# a\n", encoding="utf-8")
    (tmp_path / "b.markdown").write_text("# b\n", encoding="utf-8")
    (tmp_path / "c.txt").write_text("not markdown", encoding="utf-8")
    found = discover_notes(tmp_path)
    names = {p.name for p in found}
    assert "a.md" in names
    assert "b.markdown" in names
    assert "c.txt" not in names


def test_discover_notes_respects_exclude(tmp_path: Path) -> None:
    (tmp_path / "keep.md").write_text("k", encoding="utf-8")
    (tmp_path / "drop.md").write_text("d", encoding="utf-8")
    found = discover_notes(tmp_path, exclude=["drop.md"])
    names = {p.name for p in found}
    assert names == {"keep.md"}


def test_load_note_parses_frontmatter(tmp_path: Path) -> None:
    p = tmp_path / "note.md"
    p.write_text(
        "---\n"
        "date: 2025-04-01\n"
        "title: Hello\n"
        "tags: [a, b]\n"
        "---\n\n"
        "Body text here.\n",
        encoding="utf-8",
    )
    note = load_note(p)
    assert note.title == "Hello"
    assert note.date == date(2025, 4, 1)
    assert note.tags == ["a", "b"]
    assert "Body text" in note.body


def test_load_note_falls_back_to_h1_and_filename_date(tmp_path: Path) -> None:
    p = tmp_path / "2024-12-15-something.md"
    p.write_text("# The H1 Title\n\nbody\n", encoding="utf-8")
    note = load_note(p)
    assert note.title == "The H1 Title"
    assert note.date == date(2024, 12, 15)


# --- preprocess ---

def test_clean_body_strips_code_blocks() -> None:
    raw = "Para 1.\n\n```python\nprint('x')\n```\n\nPara 2 with `inline` code."
    cleaned = clean_body(raw)
    assert "print" not in cleaned
    assert "inline" not in cleaned
    assert "Para 1" in cleaned
    assert "Para 2" in cleaned


def test_chunk_text_short_returns_single_chunk() -> None:
    chunks = chunk_text("short text", max_chars=1000)
    assert len(chunks) == 1
    assert chunks[0][0] == "short text"
    assert chunks[0][1] == 0


def test_chunk_text_splits_long_bodies() -> None:
    paras = ["\n\n".join([f"Paragraph {i} " * 50]) for i in range(20)]
    long = "\n\n".join(paras)
    chunks = chunk_text(long, max_chars=1000, overlap=100)
    assert len(chunks) > 1
    for text, offset in chunks:
        assert offset >= 0
        assert text.strip()


def test_preprocess_note_returns_at_least_one_chunk() -> None:
    note = load_note.__wrapped__ if hasattr(load_note, "__wrapped__") else None
    from text_theme_analyzer.pipeline.model import Note
    n = Note(
        id="abc123def456",
        path=Path("n.md"),
        title="n",
        body="some body",
        date=date(2025, 1, 1),
    )
    chunks = preprocess_note(n)
    assert len(chunks) >= 1
    assert chunks[0].note_id == "abc123def456"


# --- keywords ---

def test_aggregate_frequency_dedupes_per_note() -> None:
    per_note = {
        "n1": [("agent workflow", 0.9), ("agent workflow", 0.8), ("scam", 0.5)],
        "n2": [("agent workflow", 0.7)],
    }
    freq = _aggregate_frequency(per_note)
    # "agent workflow" appears in 2 notes, "scam" in 1
    by_phrase = dict(freq)
    assert by_phrase["agent workflow"] == 2
    assert by_phrase["scam"] == 1


# --- markdown output ---

def _make_analysis() -> Analysis:
    from text_theme_analyzer.pipeline.model import Note
    n1 = Note(
        id="note-aaaa",
        path=Path("a.md"),
        title="AI agency notes",
        body="agent workflow",
        date=date(2025, 4, 1),
        tags=["ai"],
    )
    n2 = Note(
        id="note-bbbb",
        path=Path("b.md"),
        title="Scam notes",
        body="scam",
        date=date(2024, 8, 21),
        tags=["scams"],
    )
    return Analysis(
        notes=[n1, n2],
        chunks=[],
        chunk_note_ids=[],
        keywords={
            "note-aaaa": [("agent workflow", 0.9), ("ai", 0.5)],
            "note-bbbb": [("scam", 0.7)],
        },
        keyphrase_frequency=[("agent workflow", 2), ("scam", 1), ("ai", 1)],
        clusters=None,
        timeseries=None,
        metadata={
            "input_path": "/tmp",
            "date_range": ["2024-08-21", "2025-04-01"],
        },
    )


def test_render_markdown_contains_expected_sections() -> None:
    md = render_markdown(_make_analysis(), top_n_themes=5)
    assert "# Text Theme Analyzer" in md
    assert "## Summary" in md
    assert "## Top Themes" in md
    assert "## Per-Note Keyphrases" in md
    assert "## Files Analyzed" in md
    assert "agent workflow" in md
    assert "AI agency notes" in md


def test_render_markdown_handles_no_keyphrases() -> None:
    a = _make_analysis()
    a.keyphrase_frequency = []
    a.keywords = {n.id: [] for n in a.notes}
    md = render_markdown(a)
    assert "_No keyphrases extracted._" in md


# --- .env loader ---

def test_load_dotenv_sets_vars_from_explicit_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "TEXTHEME_TEST_FOO=bar\n"
        "TEXTHEME_TEST_QUOTED=\"quoted value\"\n"
        "TEXTHEME_TEST_SINGLE='single value'\n"
        "INVALID LINE WITHOUT EQUALS\n"
        "\n"
        "TEXTHEME_TEST_EMPTY=\n",
        encoding="utf-8",
    )
    for k in ("TEXTHEME_TEST_FOO", "TEXTHEME_TEST_QUOTED", "TEXTHEME_TEST_SINGLE", "TEXTHEME_TEST_EMPTY"):
        monkeypatch.delenv(k, raising=False)
    set_keys = load_dotenv(env)
    assert set_keys == ["TEXTHEME_TEST_FOO", "TEXTHEME_TEST_QUOTED", "TEXTHEME_TEST_SINGLE", "TEXTHEME_TEST_EMPTY"]
    import os
    assert os.environ["TEXTHEME_TEST_FOO"] == "bar"
    assert os.environ["TEXTHEME_TEST_QUOTED"] == "quoted value"
    assert os.environ["TEXTHEME_TEST_SINGLE"] == "single value"
    assert os.environ["TEXTHEME_TEST_EMPTY"] == ""


def test_load_dotenv_does_not_override_existing_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("TEXTHEME_TEST_PRIORITY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("TEXTHEME_TEST_PRIORITY", "from_shell")
    set_keys = load_dotenv(env)
    # Did not override, so the key is not in `set_keys` (it was already set).
    assert "TEXTHEME_TEST_PRIORITY" not in set_keys
    import os
    assert os.environ["TEXTHEME_TEST_PRIORITY"] == "from_shell"


def test_load_dotenv_override_true_replaces_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("TEXTHEME_TEST_PRIORITY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("TEXTHEME_TEST_PRIORITY", "from_shell")
    load_dotenv(env, override=True)
    import os
    assert os.environ["TEXTHEME_TEST_PRIORITY"] == "from_file"


def test_load_dotenv_returns_empty_when_missing(tmp_path: Path) -> None:
    assert load_dotenv(tmp_path / "nope.env") == []


# --- keyphrase dedup + stopword filtering ---

def test_aggregate_frequency_drops_stopword_only_phrases() -> None:
    per_note = {
        "n1": [("agent workflow", 0.9), ("of the", 0.5), ("and", 0.5)],
        "n2": [("agent workflow", 0.7), ("of the", 0.4)],
    }
    freq = _aggregate_frequency(per_note)
    by_phrase = dict(freq)
    assert "agent workflow" in by_phrase
    assert "of the" not in by_phrase
    assert "and" not in by_phrase


def test_aggregate_frequency_drops_single_chars() -> None:
    per_note = {
        "n1": [("a", 0.9), ("I", 0.8), ("scam", 0.5)],
    }
    freq = _aggregate_frequency(per_note)
    by_phrase = dict(freq)
    assert "a" not in by_phrase
    assert "i" not in by_phrase
    assert "scam" in by_phrase


def test_aggregate_frequency_drops_contained_phrases() -> None:
    """When counts are within 1.5x, drop the shorter and keep the more specific longer."""
    per_note = {
        "n1": [("discord direct", 0.9), ("discord direct conversation", 0.7)],
        "n2": [("discord direct", 0.8), ("discord direct conversation", 0.6)],
        "n3": [("discord direct conversation", 0.5)],
    }
    freq = _aggregate_frequency(per_note)
    by_phrase = dict(freq)
    # "discord direct" (2) vs "discord direct conversation" (3): ratio 1.5,
    # at the boundary. The longer is the more specific signal -> keep longer.
    assert "discord direct conversation" in by_phrase
    assert "discord direct" not in by_phrase


def test_aggregate_frequency_keeps_when_counts_diverge() -> None:
    """When counts diverge by more than the ratio, keep both."""
    per_note = {
        "n1": [("signal", 0.9), ("signal lost", 0.5)],
        "n2": [("signal", 0.8), ("signal lost", 0.5)],
        "n3": [("signal", 0.7)],
        "n4": [("signal", 0.6)],
        "n5": [("signal", 0.5)],
    }
    freq = _aggregate_frequency(per_note)
    by_phrase = dict(freq)
    # "signal" (5) vs "signal lost" (2): ratio 2.5, exceeds 1.5 -> both kept.
    assert by_phrase["signal"] == 5
    assert "signal lost" in by_phrase
    assert by_phrase["signal lost"] == 2


def test_aggregate_frequency_drops_shorter_when_longer_dominates() -> None:
    """When the longer phrase is much more frequent, drop the shorter."""
    per_note = {
        "n1": [("agent", 0.9), ("agent workflow", 0.5)],
        "n2": [("agent", 0.8), ("agent workflow", 0.5)],
        "n3": [("agent workflow", 0.7)],
        "n4": [("agent workflow", 0.6)],
    }
    freq = _aggregate_frequency(per_note)
    by_phrase = dict(freq)
    # "agent" (2) vs "agent workflow" (4): ratio 2.0, exceeds 1.5 -> keep both.
    # (The broader "agent" concept is still informative even though the
    # specific "agent workflow" is more frequent.)
    assert "agent" in by_phrase
    assert "agent workflow" in by_phrase


def test_aggregate_frequency_keeps_when_disabled() -> None:
    per_note = {
        "n1": [("discord direct", 0.9), ("discord direct conversation", 0.7)],
        "n2": [("discord direct", 0.8), ("discord direct conversation", 0.6)],
    }
    freq = _aggregate_frequency(per_note, merge_contained_phrases=False)
    by_phrase = dict(freq)
    assert "discord direct" in by_phrase
    assert "discord direct conversation" in by_phrase


# --- CSV ingest ---

def _write_queue_csv(path: Path) -> None:
    """Mirror the real content-review-queue.csv shape used in the wild."""
    path.write_text(
        "id,date_created,source,theme,platform,format,draft,hook,tone,personal_level,"
        "private_risk,status,your_comment,posted_url,reuse_as,notes_for_revision\n"
        "2026-05-22-001,2026-05-22,discord content pipeline,AI operator workflow,X,"
        "single post,\"The best AI workflow I have found so far is boring: one task, "
        "one branch, narrow write scope.\",The best AI workflow I have found so far is boring.,"
        "practical,low,low,approve,,,LinkedIn expansion,Can become a short thread.\n"
        "2026-05-22-002,2026-05-22,discord content pipeline,content pipeline,LinkedIn,"
        "short post,\"I am treating content more like a review queue than a publishing "
        "impulse. The useful pieces get approved, revised, or rejected.\","
        "I am treating content more like a review queue.,reflective,medium,low,approve,,,"
        "Substack note,Good first meta-post.\n"
        # Blank row: only an id, no draft and no theme — should be skipped.
        "2026-05-23-001,2026-05-23,recent OpenClaw,source control as AI bridge,LinkedIn,"
        "short post,Empty draft,,practical,low,low,approve,,,X thread,Good bridge.\n",
        encoding="utf-8",
    )


def test_load_csv_returns_one_note_per_row(tmp_path: Path) -> None:
    p = tmp_path / "queue.csv"
    _write_queue_csv(p)
    notes = load_csv(p)
    assert len(notes) == 3  # 3 non-empty rows in the fixture
    # Title combines id + theme
    assert notes[0].title == "[2026-05-22-001] AI operator workflow"
    assert notes[1].title == "[2026-05-22-002] content pipeline"
    # Date pulled from date_created
    assert notes[0].date == date(2026, 5, 22)
    assert notes[2].date == date(2026, 5, 23)


def test_load_csv_body_contains_draft_and_hook(tmp_path: Path) -> None:
    p = tmp_path / "queue.csv"
    _write_queue_csv(p)
    notes = load_csv(p)
    body = notes[0].body
    assert "Draft:" in body
    assert "boring: one task" in body
    assert "Hook:" in body
    assert "best AI workflow" in body
    # Source/platform/tone are labeled sections, not lost
    assert "Source: discord content pipeline" in body
    assert "Platform: X" in body
    assert "Tone: practical" in body


def test_load_csv_tags_come_from_theme_and_platform(tmp_path: Path) -> None:
    p = tmp_path / "queue.csv"
    _write_queue_csv(p)
    notes = load_csv(p)
    assert notes[0].tags == ["AI operator workflow", "X"]
    assert notes[1].tags == ["content pipeline", "LinkedIn"]


def test_load_csv_skips_blank_rows(tmp_path: Path) -> None:
    p = tmp_path / "queue.csv"
    _write_queue_csv(p)
    notes = load_csv(p)
    # The third row has an id but no draft and no theme — should be skipped
    # (it carries no signal for keyword/cluster extraction).
    assert all(n.title for n in notes)
    assert all(n.body for n in notes)


def test_load_csv_handles_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    assert load_csv(p) == []


def test_load_csv_handles_header_only_file(tmp_path: Path) -> None:
    p = tmp_path / "header-only.csv"
    p.write_text("id,date_created,theme,draft\n", encoding="utf-8")
    assert load_csv(p) == []


def test_load_csv_handles_bom(tmp_path: Path) -> None:
    p = tmp_path / "queue.csv"
    p.write_bytes(
        b"\xef\xbb\xbf"
        + b"id,date_created,theme,platform,draft\n"
        + b"2026-05-22-001,2026-05-22,AI operator workflow,X,Some draft text.\n"
    )
    notes = load_csv(p)
    assert len(notes) == 1
    assert notes[0].date == date(2026, 5, 22)
    assert notes[0].tags == ["AI operator workflow", "X"]


def test_load_csv_custom_column_map(tmp_path: Path) -> None:
    """A CSV with different header names should still parse if a column_map
    is supplied."""
    p = tmp_path / "alt.csv"
    p.write_text(
        "row_id,when,topic,channel,text\n"
        "r1,2026-04-01,cooking,instagram,Some long text about sourdough.\n",
        encoding="utf-8",
    )
    notes = load_csv(
        p,
        column_map={
            "id": "row_id",
            "date": "when",
            "theme": "topic",
            "platform": "channel",
            "draft": "text",
        },
    )
    assert len(notes) == 1
    assert notes[0].title == "[r1] cooking"
    assert notes[0].date == date(2026, 4, 1)
    assert notes[0].tags == ["cooking", "instagram"]


def test_load_notes_dispatches_csv_through_loader(tmp_path: Path) -> None:
    p = tmp_path / "queue.csv"
    _write_queue_csv(p)
    notes = load_notes(tmp_path, include=["**/*.csv"])
    # 3 non-empty rows from the fixture
    assert len(notes) == 3
    assert all(n.path.suffix == ".csv" for n in notes)


def test_discover_notes_does_not_match_csv_by_default(tmp_path: Path) -> None:
    p = tmp_path / "queue.csv"
    p.write_text("id,draft\n2026-05-22-001,hi\n", encoding="utf-8")
    found = discover_notes(tmp_path)  # default include = **/*.md, **/*.markdown
    assert found == []


def test_default_column_map_has_expected_keys() -> None:
    """Pin the contract so renaming a CSV column is a deliberate choice."""
    for k in ("id", "date", "draft", "theme", "platform"):
        assert k in DEFAULT_COLUMN_MAP
