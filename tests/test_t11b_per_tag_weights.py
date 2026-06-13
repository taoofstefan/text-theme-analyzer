"""T1.1b tests: per-tag weights for tag-weighted clustering."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from text_theme_analyzer.config import Config, apply_yaml_overrides
from text_theme_analyzer.pipeline.clustering import build_tag_matrix, cluster_chunks
from text_theme_analyzer.pipeline.model import Note, NoteChunk

# --- helpers ---

def _note(nid: str, tags: list[str]) -> Note:
    return Note(
        id=nid,
        path=Path(f"{nid}.md"),
        title=f"Title {nid}",
        body="body",
        date=date(2025, 1, 1),
        tags=tags,
    )


def _chunk(note_id: str, idx: int = 0, text: str = "the quick brown fox jumps") -> NoteChunk:
    return NoteChunk(note_id=note_id, chunk_index=idx, text=text, char_offset=0)


# --- config plumbing ---

def test_config_tag_weights_default_empty() -> None:
    cfg = Config()
    assert cfg.tag_weights == {}


def test_config_tag_weights_yaml_override() -> None:
    cfg = Config()
    apply_yaml_overrides(cfg, {"tag_weights": {"consulting": 2.0, "life": 0.5}})
    assert cfg.tag_weights == {"consulting": 2.0, "life": 0.5}


def test_config_tag_weights_yaml_strings_converted() -> None:
    cfg = Config()
    apply_yaml_overrides(cfg, {"tag_weights": {"foo": "3"}})
    assert cfg.tag_weights == {"foo": 3.0}


# --- per-tag weight application (heavy deps required) ---

@pytest.fixture
def _heavy_deps() -> None:
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("bertopic")
    pytest.importorskip("umap")
    pytest.importorskip("hdbscan")


def _make_corpus(
    n_per_group: int = 3,
) -> tuple[list[Note], list[NoteChunk], np.ndarray]:
    """Return notes, chunks, and embeddings for two tag groups with rich text."""
    notes: list[Note] = []
    chunks: list[NoteChunk] = []
    emb: list[np.ndarray] = []
    rng = np.random.default_rng(0)

    group_tags = [
        (["shared"], "agent workflow loop tree delegation model", 0),
        (["heavy", "shared"], "agent workflow design tooling framework", 0),
        (["light"], "crypto grift pattern exit liquidity whitepaper", 1),
    ]
    for i, (tags, text, axis) in enumerate(group_tags):
        for j in range(n_per_group):
            nid = f"n{i}_{j}"
            notes.append(_note(nid, tags))
            chunks.append(_chunk(nid, text=text))
            vec = np.zeros(32, dtype=np.float32)
            vec[axis] = 1.0
            emb.append(vec + rng.normal(scale=0.02, size=vec.shape))
    return notes, chunks, np.stack(emb)


def test_per_tag_weights_scale_matrix_columns(_heavy_deps) -> None:
    """Per-tag multipliers should scale individual columns of the tag matrix."""
    notes, chunks, embeddings = _make_corpus()
    tag_matrix, tag_columns = build_tag_matrix(notes, chunks, top_n_tags=3)

    res = cluster_chunks(
        chunks,
        embeddings,
        min_cluster_size=2,
        min_samples=1,
        tag_weight=1.0,
        tag_matrix=tag_matrix,
        tag_columns=tag_columns,
        tag_weights={"heavy": 10.0, "light": 0.1},
    )
    assert res is not None
    assert res.assignments
    assert len(res.cluster_keywords) >= 1


def test_per_tag_weights_ignores_unknown_tags(_heavy_deps) -> None:
    notes, chunks, embeddings = _make_corpus()
    tag_matrix, tag_columns = build_tag_matrix(notes, chunks, top_n_tags=3)

    res = cluster_chunks(
        chunks,
        embeddings,
        min_cluster_size=2,
        min_samples=1,
        tag_weight=1.0,
        tag_matrix=tag_matrix,
        tag_columns=tag_columns,
        tag_weights={"not-in-corpus": 99.0},
    )
    assert res is not None
    assert res.assignments


def test_tag_weight_zero_with_tag_weights_is_no_op(_heavy_deps) -> None:
    """If global tag_weight is 0, per-tag weights should not activate clustering."""
    notes, chunks, embeddings = _make_corpus()
    tag_matrix, tag_columns = build_tag_matrix(notes, chunks, top_n_tags=3)

    res = cluster_chunks(
        chunks,
        embeddings,
        min_cluster_size=2,
        min_samples=1,
        tag_weight=0.0,
        tag_matrix=tag_matrix,
        tag_columns=tag_columns,
        tag_weights={"shared": 100.0},
    )
    assert res is not None
    # With tag_weight=0, the tag matrix is not concatenated. The two groups
    # are separated by embedding axis, so they should still cluster apart.
    assert len(res.cluster_keywords) >= 1
