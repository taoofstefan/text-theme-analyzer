"""M2 tests: embedding cache, timeseries spike/stale, clustering wrapper.

Heavy deps (sentence-transformers, BERTopic) are not loaded here — we
test the cache mechanics and the time-series logic in isolation.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from text_theme_analyzer.pipeline.embeddings import EmbeddingCache
from text_theme_analyzer.pipeline.timeseries import _bucket_start, build_timeseries

# --- EmbeddingCache ---

def test_embedding_cache_miss_then_hit(tmp_path: Path) -> None:
    cache = EmbeddingCache(root=tmp_path, model_name="test-model")
    text = "the quick brown fox"
    assert cache.get(text) is None
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    cache.put(text, vec)
    out = cache.get(text)
    assert out is not None
    np.testing.assert_array_equal(out, vec)


def test_embedding_cache_normalizes_whitespace(tmp_path: Path) -> None:
    """Trivial whitespace edits shouldn't bust the cache."""
    cache = EmbeddingCache(root=tmp_path, model_name="test-model")
    v = np.array([1.0, 0.0], dtype=np.float32)
    cache.put("hello world", v)
    # The cache key is on the normalized (lowercased, whitespace-collapsed) form.
    assert cache.get("hello   world") is not None
    assert cache.get("HELLO WORLD") is not None


def test_embedding_cache_sharded_layout(tmp_path: Path) -> None:
    cache = EmbeddingCache(root=tmp_path, model_name="test-model")
    v = np.zeros(4, dtype=np.float32)
    cache.put("test", v)
    # The path should include a 2-char shard directory.
    files = list(tmp_path.rglob("*.npy"))
    assert len(files) == 1
    # The shard is the first 2 hex chars of the sha256 hash.
    rel = files[0].relative_to(tmp_path)
    parts = rel.parts
    # parts: ['embeddings', 'test-model', '<shard>', '<filename>.npy']
    assert len(parts) == 4
    shard = parts[-2]
    assert len(shard) == 2
    int(shard, 16)  # parses as hex


def test_embedding_cache_load_all_marks_missing(tmp_path: Path) -> None:
    cache = EmbeddingCache(root=tmp_path, model_name="t")
    v = np.array([1.0], dtype=np.float32)
    cache.put("cached text", v)
    cached, missing = cache.load_all(["cached text", "not cached"])
    assert cached[0] is not None
    assert cached[1] is None
    assert missing == [1]


# --- timeseries ---

def test_bucket_start_week_uses_monday() -> None:
    # 2024-08-15 is a Thursday; should roll back to 2024-08-12 (Monday).
    assert _bucket_start(date(2024, 8, 15), "week") == date(2024, 8, 12)


def test_bucket_start_month_truncates_day() -> None:
    assert _bucket_start(date(2024, 8, 15), "month") == date(2024, 8, 1)


def test_build_timeseries_no_data_returns_empty() -> None:
    ts = build_timeseries({}, {})
    assert ts.spikes == []
    assert ts.stale == []
    assert ts.series == {}


def test_build_timeseries_detects_spike() -> None:
    # Cluster 0: 1 per week for 8 weeks, then 5 in one week = spike.
    base = date(2024, 1, 1)  # Monday
    note_to_cluster = {}
    note_dates = {}
    nid = 0
    from datetime import timedelta
    for w in range(8):
        note_to_cluster[f"n{nid}"] = 0
        note_dates[f"n{nid}"] = base + timedelta(weeks=w)
        nid += 1
    for _ in range(5):
        note_to_cluster[f"n{nid}"] = 0
        note_dates[f"n{nid}"] = base + timedelta(weeks=9)
        nid += 1
    ts = build_timeseries(note_to_cluster, note_dates, bucket="week", spike_window=4, stale_window=4)
    assert any(s.count == 5 for s in ts.spikes)


def test_build_timeseries_detects_stale() -> None:
    # Cluster 1: 4 occurrences early, then 8 weeks of nothing -> stale.
    note_to_cluster = {}
    note_dates = {}
    base = date(2024, 1, 1)
    from datetime import timedelta
    for w in range(4):
        note_to_cluster[f"n{w}"] = 1
        note_dates[f"n{w}"] = base + timedelta(weeks=w)
    # Add a dated note for a different cluster 12 weeks later so the time axis extends.
    note_to_cluster["anchor"] = 99
    note_dates["anchor"] = base + timedelta(weeks=12)
    ts = build_timeseries(note_to_cluster, note_dates, bucket="week", stale_window=4)
    assert any(s.cluster_id == 1 for s in ts.stale)
    s = next(s for s in ts.stale if s.cluster_id == 1)
    assert s.frequency == 4


def test_build_timeseries_skips_undated_notes() -> None:
    note_to_cluster = {"n1": 0, "n2": 0}
    note_dates = {"n1": date(2024, 1, 1), "n2": None}
    ts = build_timeseries(note_to_cluster, note_dates, bucket="week")
    # Only one dated note -> series exists but no spike (only 1 occurrence).
    assert 0 in ts.series
    assert sum(ts.series[0].values()) == 1


# --- severity ladder ---

def _stale_with(total: int, last_seen_weeks_ago: int, *, anchor_weeks: int = 30,
                stale_window: int = 4) -> tuple[dict, dict]:
    """Build a synthetic note_to_cluster + note_dates pair for severity tests.

    `last_seen_weeks_ago` is the gap between the cluster's last activity and
    the right edge of the time axis.
    """
    note_to_cluster: dict[str, int] = {}
    note_dates: dict[str, date] = {}
    base = date(2024, 1, 1)
    # Spread `total` notes in the first `anchor_weeks - last_seen_weeks_ago` weeks.
    early_window = max(1, anchor_weeks - last_seen_weeks_ago)
    for w in range(min(total, early_window)):
        note_to_cluster[f"e{w}"] = 7
        note_dates[f"e{w}"] = base + timedelta(weeks=w)
    # If we need more notes than the early window, just stack them on the same week.
    extras = total - early_window
    for w in range(extras):
        note_to_cluster[f"x{w}"] = 7
        note_dates[f"x{w}"] = base + timedelta(weeks=0)
    # Anchor note (any other cluster) to extend the time axis to anchor_weeks.
    note_to_cluster["anchor"] = 99
    note_dates["anchor"] = base + timedelta(weeks=anchor_weeks)
    return note_to_cluster, note_dates


def test_severity_strong_when_frequent_and_in_first_half() -> None:
    n2c, nd = _stale_with(total=6, last_seen_weeks_ago=20)
    ts = build_timeseries(n2c, nd, bucket="week", stale_window=4)
    s = next(s for s in ts.stale if s.cluster_id == 7)
    assert s.severity == "strong"
    assert s.frequency == 6
    assert s.quiet_streak_buckets >= 4


def test_severity_medium_when_frequent_but_recent() -> None:
    # 4 notes (below strong's 5-note floor) but enough to clear medium's
    # 3-note floor. Last activity is past the midpoint of the data range.
    n2c, nd = _stale_with(total=4, last_seen_weeks_ago=4, anchor_weeks=30, stale_window=4)
    ts = build_timeseries(n2c, nd, bucket="week", stale_window=4)
    s = next(s for s in ts.stale if s.cluster_id == 7)
    assert s.severity == "medium"


def test_severity_weak_when_low_frequency_long_silence() -> None:
    # 2 notes, very long silence (>2× stale_window).
    n2c, nd = _stale_with(total=2, last_seen_weeks_ago=20, anchor_weeks=30, stale_window=4)
    ts = build_timeseries(n2c, nd, bucket="week", stale_window=4)
    s = next(s for s in ts.stale if s.cluster_id == 7)
    assert s.severity == "weak"


def test_severity_medium_for_three_notes_one_window() -> None:
    n2c, nd = _stale_with(total=3, last_seen_weeks_ago=6, anchor_weeks=20, stale_window=4)
    ts = build_timeseries(n2c, nd, bucket="week", stale_window=4)
    s = next(s for s in ts.stale if s.cluster_id == 7)
    assert s.severity == "medium"


def test_quiet_streak_reports_consecutive_zero_buckets() -> None:
    n2c, nd = _stale_with(total=3, last_seen_weeks_ago=10, anchor_weeks=20, stale_window=4)
    ts = build_timeseries(n2c, nd, bucket="week", stale_window=4)
    s = next(s for s in ts.stale if s.cluster_id == 7)
    # Silence is at least the stale_window (4 buckets).
    assert s.quiet_streak_buckets >= 4


# --- integration smoke test (only runs if all heavy deps installed) ---

def test_cluster_chunks_smoke(tmp_path: Path) -> None:
    """Smoke test the clustering wrapper on a small synthetic corpus."""
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("bertopic")
    pytest.importorskip("umap")
    pytest.importorskip("hdbscan")
    from text_theme_analyzer.pipeline.clustering import cluster_chunks
    from text_theme_analyzer.pipeline.model import NoteChunk

    # Two clear semantic groups, plus one outlier.
    chunks = [
        NoteChunk(note_id="a1", chunk_index=0, text="the agent loop is a tree", char_offset=0),
        NoteChunk(note_id="a2", chunk_index=0, text="agent design and workflow tooling", char_offset=0),
        NoteChunk(note_id="a3", chunk_index=0, text="designing the agent workflow", char_offset=0),
        NoteChunk(note_id="b1", chunk_index=0, text="grift city online and exit liquidity", char_offset=0),
        NoteChunk(note_id="b2", chunk_index=0, text="another crypto grift noted", char_offset=0),
        NoteChunk(note_id="b3", chunk_index=0, text="the grifter resume pattern", char_offset=0),
    ]
    rng = np.random.default_rng(0)
    # 32-dim embeddings: group A on first axis, group B on second. High enough
    # that UMAP's spectral init works (n_components < n_samples).
    emb = np.zeros((6, 32), dtype=np.float32)
    emb[0:3, 0] = 1.0
    emb[3:6, 1] = 1.0
    emb += rng.normal(scale=0.05, size=emb.shape)
    res = cluster_chunks(chunks, emb, min_cluster_size=2, min_samples=1)
    # Expect 2 clusters (A and B).
    assert len(res.cluster_keywords) >= 2
    # Representative note_ids should map to chunks in either the A or B group.
    rep_ids = [nid for ids in res.cluster_representatives.values() for nid in ids]
    assert any(nid in {"a1", "a2", "a3"} for nid in rep_ids)
    assert any(nid in {"b1", "b2", "b3"} for nid in rep_ids)


# --- config: clustering knobs round-trip through YAML ---

def test_min_cluster_size_default_is_none() -> None:
    from text_theme_analyzer.config import Config
    assert Config().min_cluster_size is None
    assert Config().umap_n_neighbors is None


def test_apply_yaml_overrides_sets_cluster_knobs() -> None:
    from text_theme_analyzer.config import Config, apply_yaml_overrides
    cfg = Config()
    apply_yaml_overrides(cfg, {"min_cluster_size": 7, "umap_n_neighbors": 5})
    assert cfg.min_cluster_size == 7
    assert cfg.umap_n_neighbors == 5


def test_apply_yaml_overrides_skips_null_cluster_knobs() -> None:
    """A `min_cluster_size: null` in YAML should not clobber the default heuristic."""
    from text_theme_analyzer.config import Config, apply_yaml_overrides
    cfg = Config()
    apply_yaml_overrides(cfg, {"min_cluster_size": None})
    assert cfg.min_cluster_size is None


# --- stopword list covers CSV metadata noise ---

def test_clustering_stopword_list_includes_csv_labels() -> None:
    """The labels that csv_ingest writes into row bodies must be stopwords
    in the cluster keyword extractor, otherwise they dominate the
    cluster top-words list on mixed markdown+CSV corpora."""
    import inspect

    from text_theme_analyzer.pipeline.clustering import cluster_chunks
    src = inspect.getsource(cluster_chunks)
    # Spot-check the CSV bleed-through terms
    for w in ("draft", "low", "medium", "approve", "tone", "status"):
        assert f'"{w}"' in src, f'expected "{w}" in stopword list'
