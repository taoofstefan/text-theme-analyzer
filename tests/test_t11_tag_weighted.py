"""T1.1 tests: tag-weighted clustering and LLM tag distribution.

Covers:
- `build_tag_matrix`: shape, values, untagged corpus.
- `cluster_chunks`: tag-weight changes clustering (heavy-deps gated),
  tag-weight=0 is no-op regression guard.
- LLM bundle: per-cluster `tags` key, prompt renders `tags:` line.
- Config: tag_weight + top_n_tags round-trip through YAML + env.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from text_theme_analyzer.config import Config, apply_env_overrides, apply_yaml_overrides
from text_theme_analyzer.llm.enrichment import _cluster_tags, build_bundle
from text_theme_analyzer.llm.prompts import build_user_prompt
from text_theme_analyzer.pipeline.clustering import build_tag_matrix
from text_theme_analyzer.pipeline.model import (
    Analysis,
    ClusterResult,
    Note,
    NoteChunk,
)


# --- helpers ---

def _make_note(nid: str, tags: list[str], body: str = "some body text here") -> Note:
    return Note(
        id=nid,
        path=Path(f"{nid}.md"),
        title=f"Title {nid}",
        body=body,
        date=date(2025, 1, 1),
        tags=tags,
    )


def _make_chunk(note_id: str, idx: int = 0, text: str = "the quick brown fox jumps") -> NoteChunk:
    return NoteChunk(note_id=note_id, chunk_index=idx, text=text, char_offset=0)


# --- build_tag_matrix ---

def test_tag_matrix_shape_and_values() -> None:
    """Top-N tags are reflected in 1-positions; tags outside top-N are ignored."""
    notes = [
        _make_note("n1", ["ai", "agents", "rare"]),
        _make_note("n2", ["ai", "consulting"]),
        _make_note("n3", ["consulting", "rare"]),
        _make_note("n4", ["ai", "consulting", "agents"]),
        _make_note("n5", ["once-only-tag"]),  # below the top-N cutoff
    ]
    chunks = [_make_chunk(n.id) for n in notes]
    # top_n=3 means we keep the 3 most-frequent tags: ai (3), consulting (3),
    # agents (2). rare (2) and once-only-tag (1) are dropped.
    mat, tag_columns = build_tag_matrix(notes, chunks, top_n_tags=3)
    assert mat.shape == (5, 3)
    assert mat.dtype == np.float32
    # Each row is a 0/1 vector. Verify per-note by tag name.
    notes_by_id = {n.id: n for n in notes}
    for i, note in enumerate(notes):
        row_set = set(j for j, v in enumerate(mat[i]) if v > 0)
        expected = {mat_index for mat_index, tag in enumerate(tag_columns) if tag in note.tags}
        assert row_set == expected, f"note {note.id}: expected {expected}, got {row_set}"


def _top_tags(notes: list[Note], n: int) -> list[str]:
    from collections import Counter
    c: Counter[str] = Counter()
    for note in notes:
        c.update(note.tags)
    return [t for t, _ in c.most_common(n)]


def test_tag_matrix_handles_no_tags() -> None:
    """An untagged corpus returns an (M, 0) matrix and an empty columns list."""
    notes = [_make_note(f"n{i}", []) for i in range(4)]
    chunks = [_make_chunk(n.id) for n in notes]
    mat, tag_columns = build_tag_matrix(notes, chunks, top_n_tags=20)
    assert mat.shape == (4, 0)
    assert mat.dtype == np.float32
    assert tag_columns == []
    # Critically, hstacking an (M, 0) matrix with a real (M, D) array must work.
    fake_emb = np.ones((4, 8), dtype=np.float32)
    combined = np.hstack([fake_emb, mat * 0.5])
    assert combined.shape == (4, 8)


def test_tag_matrix_handles_empty_corpus() -> None:
    """Zero notes -> (0, 0) matrix and an empty columns list."""
    mat, tag_columns = build_tag_matrix([], [], top_n_tags=10)
    assert mat.shape == (0, 0)
    assert tag_columns == []


# --- tag_columns contract (T1.1a) ---

def test_tag_matrix_returns_columns_in_frequency_order() -> None:
    """The exposed `tag_columns` mirrors `Counter.most_common(top_n_tags)`."""
    # ai: 3, consulting: 3 (tie with ai; first-seen wins per Counter contract),
    # agents: 2, rare: 2, once-only-tag: 1. With top_n=3, the first three in
    # most_common's order are ai, consulting, agents.
    notes = [
        _make_note("n1", ["ai", "agents", "rare"]),
        _make_note("n2", ["ai", "consulting"]),
        _make_note("n3", ["consulting", "rare"]),
        _make_note("n4", ["ai", "consulting", "agents"]),
        _make_note("n5", ["once-only-tag"]),
    ]
    chunks = [_make_chunk(n.id) for n in notes]
    mat, tag_columns = build_tag_matrix(notes, chunks, top_n_tags=3)
    assert tag_columns[:2] == ["ai", "consulting"]  # tied on count, ai seen first
    assert tag_columns[2] == "agents"
    assert len(tag_columns) == 3
    assert mat.shape == (5, 3)


def test_tag_matrix_columns_match_matrix_width() -> None:
    """Invariant: len(tag_columns) == matrix.shape[1] for any (notes, chunks)."""
    cases = [
        ([_make_note("n1", ["a", "b"]), _make_note("n2", ["a"])],
         [_make_chunk("n1"), _make_chunk("n2")]),
        ([_make_note("n1", [])], [_make_chunk("n1")]),
        ([], []),
        ([_make_note("n1", ["a", "b", "c", "d", "e"])], [_make_chunk("n1")]),
    ]
    for notes, chunks in cases:
        mat, tag_columns = build_tag_matrix(notes, chunks, top_n_tags=2)
        assert len(tag_columns) == mat.shape[1], (
            f"columns/width mismatch for {len(notes)} notes: "
            f"len({tag_columns})={len(tag_columns)} vs shape[1]={mat.shape[1]}"
        )


def test_tag_matrix_columns_are_deduped() -> None:
    """Even when notes carry the same tag in many positions, columns are unique.

    Pins the Counter.most_common contract: it returns each key exactly once.
    """
    notes = [
        _make_note("n1", ["ai", "ai"]),  # duplicate within a single note
        _make_note("n2", ["ai", "consulting"]),
        _make_note("n3", ["ai"]),
    ]
    chunks = [_make_chunk(n.id) for n in notes]
    mat, tag_columns = build_tag_matrix(notes, chunks, top_n_tags=10)
    assert tag_columns == ["ai", "consulting"]
    assert mat.shape == (3, 2)
    # Every column in the matrix has at least one 1.0 (ai should be in all 3).
    assert mat[:, 0].sum() == 3.0
    assert mat[:, 1].sum() == 1.0


def test_tag_matrix_columns_align_to_matrix_indices() -> None:
    """The hook the future `tag_weights` config needs: column j is tag_columns[j]."""
    notes = [
        _make_note("n1", ["rare", "common"]),  # common wins on freq
        _make_note("n2", ["common"]),
        _make_note("n3", ["common", "rare"]),
    ]
    chunks = [_make_chunk(n.id) for n in notes]
    mat, tag_columns = build_tag_matrix(notes, chunks, top_n_tags=2)
    assert tag_columns == ["common", "rare"]
    # n1 has both tags -> both columns 1.0
    # n2 has only common -> column 0 = 1.0, column 1 = 0.0
    # n3 has both tags -> both columns 1.0
    assert mat[0].tolist() == [1.0, 1.0]
    assert mat[1].tolist() == [1.0, 0.0]
    assert mat[2].tolist() == [1.0, 1.0]


# --- cluster_chunks with tag_weight ---

def test_cluster_chunks_tag_weight_zero_is_default_behavior() -> None:
    """tag_weight=0 and no tag_matrix must be a no-op (regression guard)."""
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("bertopic")
    pytest.importorskip("umap")
    pytest.importorskip("hdbscan")
    from text_theme_analyzer.pipeline.clustering import cluster_chunks

    # Two clear semantic groups, no tags. Each chunk has substantive text
    # (BERTopic's CountVectorizer filters stopwords, so single-word bodies
    # would leave no terms and crash — keep the text rich).
    chunks = [
        _make_chunk("a1", text="agent workflow loop tree delegation model"),
        _make_chunk("a2", text="designing agent workflow orchestration"),
        _make_chunk("a3", text="agent loop model framework design"),
        _make_chunk("b1", text="grift pattern crypto anonymous team whitepaper"),
        _make_chunk("b2", text="another crypto grift noted pattern"),
        _make_chunk("b3", text="grifter resume pattern crypto scam"),
    ]
    rng = np.random.default_rng(0)
    emb = np.zeros((6, 32), dtype=np.float32)
    emb[0:3, 0] = 1.0
    emb[3:6, 1] = 1.0
    emb += rng.normal(scale=0.05, size=emb.shape)

    # Baseline (no tag weighting).
    res_default = cluster_chunks(chunks, emb, min_cluster_size=2, min_samples=1)
    # tag_weight=0 with no tag_matrix (default kwarg).
    res_zero = cluster_chunks(chunks, emb, min_cluster_size=2, min_samples=1, tag_weight=0.0)
    # tag_weight=0.5 with an empty tag_matrix -> still a no-op.
    res_zero_with_matrix = cluster_chunks(
        chunks, emb, min_cluster_size=2, min_samples=1,
        tag_weight=0.5, tag_matrix=np.zeros((6, 0), dtype=np.float32),
    )
    # All three should yield the same cluster assignments and umap_2d.
    assert res_default.assignments == res_zero.assignments
    assert res_default.assignments == res_zero_with_matrix.assignments
    assert res_default.umap_2d == res_zero.umap_2d


def test_cluster_chunks_with_tag_weight_changes_clustering() -> None:
    """Synthetic corpus where tag overlap overrides embedding similarity.

    Embedding layout: A and B are close (group 1), C is far (group 2).
    Tag layout: A and C share a tag, B has a different tag.

    With tag_weight=0, the embedding dominates -> {A,B} cluster, {C} cluster.
    With tag_weight=high, the tag signal pulls A and C together.
    """
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("bertopic")
    pytest.importorskip("umap")
    pytest.importorskip("hdbscan")
    from text_theme_analyzer.pipeline.clustering import cluster_chunks

    notes = [
        _make_note("a1", ["shared"]),
        _make_note("a2", ["shared"]),
        _make_note("b1", ["different"]),
        _make_note("b2", ["different"]),
        _make_note("c1", ["shared"]),
        _make_note("c2", ["shared"]),
    ]
    # Each chunk has substantive text. Group A and B share the "agent" theme
    # in text (so the embedding-based clustering prefers {A,B} together),
    # group C talks about "career" and "transition" (semantically distinct).
    chunks = [
        _make_chunk("a1", text="agent workflow loop tree delegation model"),
        _make_chunk("a2", text="designing agent workflow orchestration"),
        _make_chunk("b1", text="agent pattern design framework model"),
        _make_chunk("b2", text="another agent framework design noted"),
        _make_chunk("c1", text="career transition freelance consulting work"),
        _make_chunk("c2", text="career consulting freelance transition plan"),
    ]
    # Embedding: A and B are close, C is far. Tag-space: A and C close, B far.
    emb = np.zeros((6, 32), dtype=np.float32)
    emb[0:2, 0] = 1.0   # A
    emb[2:4, 0] = 1.0   # B (close to A in embedding)
    emb[4:6, 1] = 1.0   # C (far in embedding)
    emb += np.random.default_rng(0).normal(scale=0.02, size=emb.shape)
    notes_by_id = {n.id: n for n in notes}
    tag_matrix, _tag_columns = build_tag_matrix(notes, chunks, top_n_tags=2)

    # Baseline: A and B cluster, C is separate (because embedding distance
    # puts them together).
    res_default = cluster_chunks(chunks, emb, min_cluster_size=2, min_samples=1)
    # High tag weight: A and C cluster (because they share tags).
    res_tagged = cluster_chunks(
        chunks, emb, min_cluster_size=2, min_samples=1,
        tag_weight=10.0, tag_matrix=tag_matrix,
    )
    # The two results should be observably different. Either the cluster
    # assignments change, or the number of clusters/outliers changes. We
    # don't pin to a specific outcome (UMAP/HDBSCAN are stochastic in
    # general) — we just require the *structural* result to differ.
    same = (
        res_default.assignments == res_tagged.assignments
        and res_default.cluster_sizes == res_tagged.cluster_sizes
    )
    assert not same, (
        "tag_weight=10 had no effect on clustering — the tag signal is not "
        "actually reaching the embedding"
    )


# --- _cluster_tags (LLM bundle helper) ---

def _fake_analysis_with_tags() -> Analysis:
    """Two-cluster analysis where clusters have distinct tag distributions."""
    n1 = _make_note("n1", ["ai", "consulting", "agents"], body="agent stuff")
    n2 = _make_note("n2", ["ai", "consulting"], body="more agent stuff")
    n3 = _make_note("n3", ["grift", "crypto", "scams"], body="grift stuff")
    n4 = _make_note("n4", ["grift", "crypto"], body="more grift stuff")
    chunks = [_make_chunk(n.id) for n in (n1, n2, n3, n4)]
    cluster_result = ClusterResult(
        assignments=[0, 0, 1, 1],
        cluster_sizes={0: 2, 1: 2},
        cluster_keywords={0: [("agent", 0.3)], 1: [("grift", 0.3)]},
        cluster_representatives={0: ["n1"], 1: ["n3"]},
        umap_2d=[(0, 0), (0.1, 0.1), (1, 1), (1.1, 1.1)],
        outlier_count=0,
    )
    return Analysis(
        notes=[n1, n2, n3, n4],
        chunks=chunks,
        chunk_note_ids=["n1", "n2", "n3", "n4"],
        keywords={n.id: [] for n in (n1, n2, n3, n4)},
        keyphrase_frequency=[],
        clusters=cluster_result,
        timeseries=None,
        enrichment=None,
        metadata={},
    )


def test_cluster_tags_returns_per_cluster_distribution() -> None:
    a = _fake_analysis_with_tags()
    tags = _cluster_tags(a, top_n=20)
    assert 0 in tags and 1 in tags
    # Cluster 0 has notes n1, n2. Tags: n1={ai, consulting, agents}, n2={ai, consulting}.
    # Counts: ai=2, consulting=2, agents=1. (ai vs consulting tie on count;
    # Counter preserves insertion order from the first note, but the
    # dedup-via-set() in our implementation makes that nondeterministic.
    # The contract: top of list is one of the tied pair, and the singleton
    # "agents" comes last.)
    set0 = set(tags[0])
    assert set0 == {"ai", "consulting", "agents"}
    assert tags[0][-1] == "agents", (
        f"expected singleton tag last, got order {tags[0]}"
    )
    # Cluster 1 has notes n3, n4. Tags: n3={grift, crypto, scams}, n4={grift, crypto}.
    # Counts: grift=2, crypto=2, scams=1. Same tie-breaking caveat.
    set1 = set(tags[1])
    assert set1 == {"grift", "crypto", "scams"}
    assert tags[1][-1] == "scams"


def test_cluster_tags_empty_for_untagged_corpus() -> None:
    notes = [_make_note(f"n{i}", []) for i in range(3)]
    chunks = [_make_chunk(n.id) for n in notes]
    a = Analysis(
        notes=notes, chunks=chunks, chunk_note_ids=[n.id for n in notes],
        keywords={}, keyphrase_frequency=[],
        clusters=ClusterResult(
            assignments=[0, 0, 0], cluster_sizes={0: 3},
            cluster_keywords={0: [("foo", 0.3)]},
            cluster_representatives={0: ["n0"]}, umap_2d=[(0, 0)] * 3, outlier_count=0,
        ),
        timeseries=None, enrichment=None, metadata={},
    )
    assert _cluster_tags(a) == {}


# --- build_bundle includes tags ---

def test_build_bundle_includes_tags_per_cluster() -> None:
    a = _fake_analysis_with_tags()
    bundle = build_bundle(a)
    for c in bundle["clusters"]:
        assert "tags" in c
        assert isinstance(c["tags"], list)
        # Every cluster here is tagged, so the list is non-empty.
        assert len(c["tags"]) > 0


# --- prompt renders tags line ---

def test_prompt_renders_tags_line() -> None:
    """The user prompt must surface the per-cluster tag list."""
    user = build_user_prompt(
        total_notes=4,
        date_range=("2025-01-01", "2025-01-02"),
        clusters=[{
            "id": 0, "size": 2,
            "keywords": ["agent"],
            "keyphrases": ["agent"],
            "tags": ["ai", "consulting", "agents"],
            "representative_titles": ["t"],
            "representative_quotes": ["q"],
            "excerpts": [{"title": "t", "date": "2025-01-01", "body": "body"}],
            "first_seen": "2025-01-01", "last_seen": "2025-01-01",
        }],
        spikes=[],
        stale_candidates=[],
    )
    assert "tags: ai, consulting, agents" in user


def test_prompt_renders_no_tags_fallback() -> None:
    """When a cluster has no tags, the line says (none) so the LLM doesn't
    infer missing data."""
    user = build_user_prompt(
        total_notes=1,
        date_range=(None, None),
        clusters=[{
            "id": 0, "size": 1,
            "keywords": ["agent"],
            "keyphrases": [],
            "tags": [],
            "representative_titles": ["t"],
            "representative_quotes": [],
            "excerpts": [],
            "first_seen": "?", "last_seen": "?",
        }],
        spikes=[],
        stale_candidates=[],
    )
    assert "tags: (none)" in user


# --- config round-trip ---

def test_config_tag_weight_defaults() -> None:
    cfg = Config()
    assert cfg.tag_weight == 0.0
    assert cfg.top_n_tags == 20
    assert cfg.tag_field == "both"


def test_config_tag_weight_yaml_override() -> None:
    cfg = Config()
    apply_yaml_overrides(cfg, {"tag_weight": 0.3, "top_n_tags": 10})
    assert cfg.tag_weight == 0.3
    assert cfg.top_n_tags == 10


def test_config_tag_weight_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEXTHEME_TAG_WEIGHT", "0.42")
    cfg = Config()
    apply_env_overrides(cfg)
    assert cfg.tag_weight == 0.42


def test_config_tag_weight_env_malformed_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed env var should not clobber the default — same pattern as
    TEXTHEME_OLLAMA_TIMEOUT."""
    monkeypatch.setenv("TEXTHEME_TAG_WEIGHT", "not-a-float")
    cfg = Config()
    apply_env_overrides(cfg)
    assert cfg.tag_weight == 0.0
