"""Tests for the pure-Python zero-dep keyword fallback (T1.4)."""

from __future__ import annotations

import sys

from text_theme_analyzer.pipeline.keywords import (
    extract_keyphrases,
    extract_zero_dep,
)
from text_theme_analyzer.pipeline.model import NoteChunk

# --- extract_zero_dep unit tests ---


def test_zero_dep_empty_text() -> None:
    assert extract_zero_dep([""], top_n=5) == [[]]
    assert extract_zero_dep(["   \n\t  "], top_n=5) == [[]]


def test_zero_dep_stopwords_only() -> None:
    assert extract_zero_dep(["the and a of in on at to"], top_n=5) == [[]]


def test_zero_dep_extracts_phrases() -> None:
    texts = [
        "The agent design loop is a tree of decisions.",
        "Agent design and workflow tooling for productive teams.",
        "Crypto grift patterns and exit liquidity are everywhere.",
    ]
    out = extract_zero_dep(texts, top_n=10)
    assert len(out) == 3
    for phrases in out:
        assert phrases
        assert all(isinstance(p, str) and isinstance(s, (int, float)) for p, s in phrases)
        assert len(phrases) <= 10
    flat = [p for phrases in out for p, _ in phrases]
    # Multi-word phrases should be present, and content words from each doc.
    assert any(" " in p for p in flat)
    assert any("grift" in p for p in flat)


def test_zero_dep_idf_prefers_rare_terms() -> None:
    """A term that appears in only one doc should score higher than one in every doc."""
    texts = [
        "alpha beta gamma delta",  # alpha in every doc -> low idf
        "beta gamma delta epsilon",  # epsilon only here -> high idf
    ]
    out = extract_zero_dep(texts, top_n=2)
    # Top phrase in the second doc should include epsilon (high idf).
    second_doc_top = out[1][0][0]
    assert "epsilon" in second_doc_top


def test_zero_dep_respects_top_n() -> None:
    text = "foo bar baz qux zot"
    out = extract_zero_dep([text], top_n=2)
    assert len(out[0]) == 2


# --- extract_keyphrases routing ---


def test_keyphrases_yake_routes_to_zero_dep() -> None:
    """The legacy method='yake' name no longer imports yake; it uses zero_dep."""
    chunks = [
        NoteChunk(note_id="n1", chunk_index=0, text="agent design workflow", char_offset=0),
    ]
    out = extract_keyphrases(chunks, method="yake", top_n=3)
    assert len(out) == 1
    assert out[0]


def test_keyphrases_keybert_falls_back_on_import_error(monkeypatch) -> None:
    """method='keybert' falls back to zero_dep when KeyBERT is unavailable."""

    def _raise_import(*_args, **_kwargs) -> None:
        raise ImportError("no keybert")

    monkeypatch.setattr(
        "text_theme_analyzer.pipeline.keywords.extract_with_keybert",
        _raise_import,
    )
    chunks = [
        NoteChunk(note_id="n1", chunk_index=0, text="agent design workflow", char_offset=0),
    ]
    out = extract_keyphrases(chunks, method="keybert", top_n=3)
    assert len(out) == 1
    assert out[0]


def test_yake_unavailable_still_routes_to_zero_dep(monkeypatch) -> None:
    """method='yake' must keep working even if the yake package is absent."""
    # Force any attempt to import yake to fail, then prove we don't need it.
    monkeypatch.setitem(sys.modules, "yake", None)
    chunks = [
        NoteChunk(note_id="n1", chunk_index=0, text="agent design workflow", char_offset=0),
    ]
    out = extract_keyphrases(chunks, method="yake", top_n=3)
    assert len(out) == 1
    assert out[0]
