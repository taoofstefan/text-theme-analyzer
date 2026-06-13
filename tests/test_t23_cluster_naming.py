"""T2.3 tests: persistent cluster names across runs."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from text_theme_analyzer.llm.schemas import ClusterAnnotation
from text_theme_analyzer.pipeline.cluster_naming import (
    CLUSTER_NAMES_FILENAME,
    build_name_catalog,
    load_name_catalog,
    merge_catalogs,
    resolve_stable_name,
    resolve_stable_names,
    save_name_catalog,
)
from text_theme_analyzer.pipeline.model import (
    Analysis,
    ClusterResult,
    Note,
    NoteChunk,
)

# --- helpers ---

def _make_analysis(
    *,
    assignments: list[int],
    embeddings: np.ndarray,
    keywords: dict[int, list[tuple[str, float]]],
    enrichment_names: dict[int, str] | None = None,
) -> Analysis:
    notes = [Note(id="n0", path=Path("n0.md"), title="N", body="body", date=date(2025, 1, 1))]
    chunks = [NoteChunk(note_id="n0", chunk_index=0, text="body", char_offset=0)]
    enrichment = None
    if enrichment_names:
        clusters = [
            ClusterAnnotation(
                cluster_id=cid,
                name=name,
                summary="summary",
                emotional_tone="calm",
            )
            for cid, name in enrichment_names.items()
        ]
        from text_theme_analyzer.llm.schemas import EnrichmentResult
        enrichment = EnrichmentResult(clusters=clusters)
    return Analysis(
        notes=notes,
        chunks=chunks,
        chunk_note_ids=["n0"],
        keywords={},
        keyphrase_frequency=[],
        clusters=ClusterResult(
            assignments=assignments,
            cluster_sizes={cid: assignments.count(cid) for cid in set(assignments) if cid != -1},
            cluster_keywords=keywords,
            cluster_representatives={},
            umap_2d=[(0.0, 0.0)] * len(assignments),
            outlier_count=sum(1 for a in assignments if a == -1),
        ),
        timeseries=None,
        enrichment=enrichment,
        metadata={},
    )


# --- catalog building ---

def test_build_name_catalog_prefers_llm_name() -> None:
    emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    analysis = _make_analysis(
        assignments=[0, 0, 1],
        embeddings=emb,
        keywords={0: [("alpha", 0.5)], 1: [("beta", 0.5)]},
        enrichment_names={0: "LLM Alpha", 1: "LLM Beta"},
    )
    catalog = build_name_catalog(analysis, emb)
    assert len(catalog) == 2
    names = {entry["name"] for entry in catalog.values()}
    assert "LLM Alpha" in names
    assert "LLM Beta" in names


def test_build_name_catalog_falls_back_to_keywords() -> None:
    emb = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    analysis = _make_analysis(
        assignments=[0, 0],
        embeddings=emb,
        keywords={0: [("alpha", 0.5), ("beta", 0.4)]},
    )
    catalog = build_name_catalog(analysis, emb)
    assert len(catalog) == 1
    assert catalog[list(catalog.keys())[0]]["name"] == "alpha / beta"


# --- save/load/merge ---

def test_save_and_load_catalog_round_trip(tmp_path: Path) -> None:
    catalog = {"abc123": {"name": "foo", "centroid": [1.0, 0.0], "last_seen_cid": 0}}
    path = save_name_catalog(tmp_path, catalog)
    assert path.name == CLUSTER_NAMES_FILENAME
    loaded = load_name_catalog(tmp_path)
    assert loaded == catalog


def test_merge_catalog_new_wins() -> None:
    old = {"k": {"name": "old", "centroid": [1.0, 0.0], "last_seen_cid": 0}}
    new = {"k": {"name": "new", "centroid": [1.0, 0.0], "last_seen_cid": 0}}
    merged = merge_catalogs(old, new)
    assert merged["k"]["name"] == "new"


# --- similarity resolution ---

def test_resolve_stable_name_matches_near_identical_centroid() -> None:
    catalog = {
        "k1": {"name": "Agent workflow", "centroid": [1.0, 0.0, 0.0]},
    }
    name = resolve_stable_name(np.array([1.0, 0.0, 0.0], dtype=np.float32), catalog, threshold=0.85)
    assert name == "Agent workflow"


def test_resolve_stable_name_returns_none_for_distant_centroid() -> None:
    catalog = {
        "k1": {"name": "Agent workflow", "centroid": [1.0, 0.0, 0.0]},
    }
    name = resolve_stable_name(np.array([0.0, 1.0, 0.0], dtype=np.float32), catalog, threshold=0.85)
    assert name is None


def test_resolve_stable_names_maps_cluster_ids() -> None:
    emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    analysis = _make_analysis(
        assignments=[0, 0, 1],
        embeddings=emb,
        keywords={0: [("alpha", 0.5)], 1: [("beta", 0.5)]},
    )
    catalog = {
        "near0": {"name": "Stable Alpha", "centroid": [1.0, 0.0]},
    }
    mapping = resolve_stable_names(analysis, emb, catalog, threshold=0.85)
    assert mapping == {0: "Stable Alpha"}


# --- orchestrator integration (CLI smoke) ---

def test_orchestrator_saves_cluster_catalog(tmp_path: Path) -> None:
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("bertopic")
    pytest.importorskip("umap")
    pytest.importorskip("hdbscan")
    from text_theme_analyzer.config import Config
    from text_theme_analyzer.pipeline.orchestrator import run

    notes = tmp_path / "notes"
    notes.mkdir()
    for i, text in enumerate([
        "agent workflow loop tree delegation model",
        "designing agent workflow orchestration",
        "agent loop model framework design",
        "crypto grift pattern exit liquidity whitepaper",
        "another crypto grift noted pattern",
        "grifter resume pattern crypto scam",
    ]):
        (notes / f"2025-01-0{i+1}.md").write_text(f"# N{i}\n\n{text}", encoding="utf-8")

    cfg = Config()
    cfg.input_path = notes
    cfg.output_dir = tmp_path / "out"
    cfg.no_llm = True
    cfg.no_cache = True
    analysis = run(cfg)
    assert analysis.clusters is not None
    assert len(analysis.clusters.cluster_keywords) >= 1
    catalog_path = cfg.output_dir / CLUSTER_NAMES_FILENAME
    assert catalog_path.is_file()
    loaded = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert len(loaded) >= 1
