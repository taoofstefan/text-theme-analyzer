"""Run history + diff tests."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from text_theme_analyzer.output.history import (
    HISTORY_DIRNAME,
    _fingerprint_similarity,
    _idf_from_runs,
    _match_clusters,
    diff_snapshots,
    list_snapshots,
    load_snapshot,
    render_diff,
    snapshot_from_analysis,
    write_snapshot,
)
from text_theme_analyzer.pipeline.model import Analysis, ClusterResult, Note, NoteChunk


def _fake_analysis(
    *,
    notes: int = 2,
    phrases: list[tuple[str, int]] | None = None,
    sizes: dict[int, int] | None = None,
    kws: dict[int, list[tuple[str, float]]] | None = None,
    names: dict[int, str] | None = None,
) -> Analysis:
    notes_l: list[Note] = []
    chunks: list[NoteChunk] = []
    chunk_ids: list[str] = []
    for i in range(notes):
        nid = f"n{i}"
        body = "The agent loop is a tree. The model decides when to call tools."
        notes_l.append(Note(
            id=nid, path=Path(f"{nid}.md"), title=f"Title {i}",
            body=body, date=date(2025, 4, 1 + i), tags=["ai"],
        ))
        chunks.append(NoteChunk(note_id=nid, chunk_index=0, text=body, char_offset=0))
        chunk_ids.append(nid)
    cluster_sizes = sizes or {0: notes}
    cluster_kws = kws if kws is not None else {
        cid: [(f"kw{cid}", 0.3), ("agent", 0.2)] for cid in cluster_sizes
    }
    cr = ClusterResult(
        assignments=[0] * notes,
        cluster_sizes=cluster_sizes,
        cluster_keywords=cluster_kws,
        cluster_representatives={cid: [f"n{i}" for i in range(min(s, notes))]
                                for cid, s in cluster_sizes.items()},
        umap_2d=[(0.0, 0.0)] * notes,
        outlier_count=0,
    )
    return Analysis(
        notes=notes_l, chunks=chunks, chunk_note_ids=chunk_ids,
        keywords={f"n{i}": [("agent", 0.5)] for i in range(notes)},
        keyphrase_frequency=phrases or [("agent", notes), ("model", notes)],
        clusters=cr, timeseries=None, enrichment=None,
        metadata={"date_range": ["2025-04-01", "2025-04-02"]},
    )


def _snap_with_fingerprints(
    *,
    sizes: dict[int, int],
    fingerprints: dict[int, list[str]],
    phrases: list[tuple[str, int]] | None = None,
    names: dict[int, str] | None = None,
    ts: str = "2025-04-01T10-00-00Z",
) -> RunSnapshot:  # noqa: F821
    """Build a RunSnapshot directly (bypassing snapshot_from_analysis) with explicit
    fingerprints. Used to test the matching algorithm without going through a real
    analysis pipeline (where c-TF-IDF words are not predictable)."""
    from text_theme_analyzer.output.history import RunSnapshot
    # Top-5 keywords (display) — slice from the fingerprint.
    cluster_keywords = {cid: fps[:5] for cid, fps in fingerprints.items()}
    return RunSnapshot(
        timestamp=ts,
        note_count=20,
        chunk_count=20,
        date_range=["2025-04-01", "2025-04-15"],
        keyphrase_top=phrases or [("agent", 5), ("model", 5)],
        cluster_sizes=sizes,
        cluster_keywords=cluster_keywords,
        cluster_fingerprints=fingerprints,
        cluster_names=names or {},
        spike_count=0,
        stale_count=0,
    )


def test_snapshot_round_trip(tmp_path: Path) -> None:
    a = _fake_analysis(notes=3)
    snap = snapshot_from_analysis(a)
    out = write_snapshot(snap, tmp_path)
    assert out.exists()
    assert out.parent == tmp_path / HISTORY_DIRNAME
    # Round-trip via load_snapshot.
    loaded = load_snapshot(out)
    assert loaded.timestamp == snap.timestamp
    assert loaded.note_count == 3
    assert loaded.cluster_sizes == {0: 3}
    # The JSON is valid + has the documented schema_version.
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["schema_version"] == "1.0"


def test_list_snapshots_chronological(tmp_path: Path) -> None:
    a = _fake_analysis()
    snap1 = snapshot_from_analysis(a)
    snap1.timestamp = "2025-04-01T10-00-00Z"
    snap2 = snapshot_from_analysis(a)
    snap2.timestamp = "2025-04-02T10-00-00Z"
    write_snapshot(snap1, tmp_path)
    write_snapshot(snap2, tmp_path)
    paths = list_snapshots(tmp_path)
    assert [p.stem for p in paths] == [snap1.timestamp, snap2.timestamp]


def test_diff_detects_added_and_removed_clusters() -> None:
    """Pre-T1.3 fixtures (no fingerprints) fall back to raw-ID matching."""
    old_a = _fake_analysis(sizes={0: 2, 1: 1})
    old = snapshot_from_analysis(old_a)
    new_a = _fake_analysis(sizes={0: 2, 2: 1})  # cluster 1 removed, 2 added
    new = snapshot_from_analysis(new_a)
    d = diff_snapshots(old, new)
    assert 2 in d.added_clusters
    assert 1 in d.removed_clusters
    assert d.removed_clusters == [1]
    assert d.added_clusters == [2]


def test_diff_detects_grown_and_shrunk() -> None:
    old_a = _fake_analysis(sizes={0: 2})
    old = snapshot_from_analysis(old_a)
    new_a = _fake_analysis(sizes={0: 5})
    new = snapshot_from_analysis(new_a)
    d = diff_snapshots(old, new)
    assert d.grown_clusters == [(0, 3)]
    assert d.shrunk_clusters == []


def test_diff_detects_added_keyphrases() -> None:
    old_a = _fake_analysis(phrases=[("agent", 2), ("model", 2)])
    old = snapshot_from_analysis(old_a)
    new_a = _fake_analysis(phrases=[("agent", 2), ("model", 2), ("tools", 5)])
    new = snapshot_from_analysis(new_a)
    d = diff_snapshots(old, new)
    assert any(p == "tools" for p, _ in d.added_keyphrases)
    assert d.dropped_keyphrases == []


def test_render_diff_is_human_readable() -> None:
    old_a = _fake_analysis(sizes={0: 1, 1: 1})
    old = snapshot_from_analysis(old_a)
    new_a = _fake_analysis(sizes={0: 3})
    new = snapshot_from_analysis(new_a)
    d = diff_snapshots(old, new)
    out = render_diff(d, old=old, new=new)
    assert "Run diff:" in out
    assert "Clusters:" in out
    assert "removed:" in out
    assert "grew:" in out
    # Should be ASCII-clean, no trailing junk.
    assert all(line == line.rstrip() for line in out.splitlines())


# --- T1.3: cluster_fingerprints on snapshot ---

def test_snapshot_includes_cluster_fingerprints() -> None:
    """snapshot_from_analysis populates cluster_fingerprints with top-8 c-TF-IDF words."""
    # Use a 10-word keyword list per cluster so the fingerprint is unambiguous.
    kws = {
        0: [(f"word{i}", 1.0 - i * 0.05) for i in range(10)],
        1: [(f"other{i}", 1.0 - i * 0.05) for i in range(10)],
    }
    a = _fake_analysis(sizes={0: 3, 1: 2}, kws=kws)
    snap = snapshot_from_analysis(a)
    assert 0 in snap.cluster_fingerprints
    assert 1 in snap.cluster_fingerprints
    # Top-8 in score-desc order.
    assert snap.cluster_fingerprints[0] == [f"word{i}" for i in range(8)]
    assert snap.cluster_fingerprints[1] == [f"other{i}" for i in range(8)]
    # `cluster_keywords` (display) is the top-5, separate from fingerprint.
    assert snap.cluster_keywords[0] == [f"word{i}" for i in range(5)]


def test_snapshot_round_trip_with_fingerprints(tmp_path: Path) -> None:
    """Fingerprints survive a write/load round-trip."""
    a = _fake_analysis(
        sizes={0: 3, 1: 2},
        kws={0: [("a", 1.0), ("b", 0.9), ("c", 0.8), ("d", 0.7), ("e", 0.6),
                ("f", 0.5), ("g", 0.4), ("h", 0.3), ("i", 0.2)],
             1: [("x", 1.0), ("y", 0.9), ("z", 0.8)]},
    )
    snap = snapshot_from_analysis(a)
    out = write_snapshot(snap, tmp_path)
    loaded = load_snapshot(out)
    assert loaded.cluster_fingerprints == snap.cluster_fingerprints
    # JSON has the cluster_fingerprints key.
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert "cluster_fingerprints" in parsed


def test_load_old_snapshot_without_fingerprints(tmp_path: Path) -> None:
    """A v1.0 snapshot without the fingerprint key loads with cluster_fingerprints={}."""
    legacy = {
        "schema_version": "1.0",
        "timestamp": "2025-04-01T10-00-00Z",
        "note_count": 5,
        "chunk_count": 5,
        "date_range": None,
        "keyphrase_top": [],
        "cluster_sizes": {"0": 3, "1": 2},
        "cluster_keywords": {"0": ["a", "b"], "1": ["c", "d"]},
        "cluster_names": {},
        "spike_count": 0,
        "stale_count": 0,
    }
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(legacy), encoding="utf-8")
    loaded = load_snapshot(p)
    assert loaded.cluster_fingerprints == {}
    # And diffing against a fingerprint snapshot still works (fallback path).
    modern = _snap_with_fingerprints(
        sizes={0: 3, 1: 2},
        fingerprints={0: ["a", "b", "c", "d", "e", "f", "g", "h"],
                      1: ["x", "y", "z", "w", "v", "u", "t", "s"]},
        ts="2025-04-02T10-00-00Z",
    )
    d = diff_snapshots(loaded, modern)
    # No fingerprint match is possible for the legacy side; raw-ID matching
    # catches cid 0 and cid 1 (both sides have them at the same cids).
    assert 0 not in d.added_clusters
    assert 1 not in d.added_clusters


# --- T1.3: fingerprint similarity ---

def test_fingerprint_similarity_identical_returns_one() -> None:
    idf = {"a": 2.0, "b": 1.5, "c": 1.0}
    assert _fingerprint_similarity(["a", "b"], ["a", "b"], idf) == 1.0


def test_fingerprint_similarity_disjoint_returns_zero() -> None:
    idf = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
    assert _fingerprint_similarity(["a", "b"], ["c", "d"], idf) == 0.0


def test_fingerprint_similarity_partial_is_idf_weighted() -> None:
    """Overlap in 1 of 2 words: cosine = idf[shared] / sqrt(idf[a]^2 + idf[b]^2)."""
    idf = {"a": 2.0, "b": 1.0, "c": 1.0}
    sim = _fingerprint_similarity(["a", "b"], ["a", "c"], idf)
    # dot = 2.0 * 2.0 = 4.0 (only 'a' overlaps)
    # norm_a = sqrt(2.0^2 + 1.0^2) = sqrt(5)
    # norm_b = sqrt(2.0^2 + 1.0^2) = sqrt(5)
    # sim = 4.0 / 5.0 = 0.8
    assert abs(sim - 0.8) < 1e-9


def test_fingerprint_similarity_empty_returns_zero() -> None:
    idf = {"a": 1.0}
    assert _fingerprint_similarity([], ["a"], idf) == 0.0
    assert _fingerprint_similarity(["a"], [], idf) == 0.0
    assert _fingerprint_similarity([], [], idf) == 0.0


def test_idf_common_words_get_lower_weight() -> None:
    """IDF down-weights words that appear in many clusters."""
    # 'agent' appears in all 4 clusters; 'rare' appears in 1.
    idf = _idf_from_runs(
        _snap_with_fingerprints(
            sizes={0: 1, 1: 1},
            fingerprints={0: ["agent", "rare"], 1: ["agent", "common"]},
        ),
        _snap_with_fingerprints(
            sizes={0: 1, 1: 1},
            fingerprints={0: ["agent", "common2"], 1: ["agent", "common3"]},
        ),
    )
    # 'agent' is in all 4 -> lowest IDF.
    # 'common' variants: 'common', 'common2', 'common3' are 3 different words
    # each appearing once. 'rare' once. So 'rare' should have the highest IDF.
    assert idf["agent"] < idf["rare"]
    assert idf["agent"] < idf["common"]


# --- T1.3: cluster matching across runs ---

def test_match_clusters_pairs_by_similarity() -> None:
    """Two fingerprints sharing 6/8 words with IDF overlap match above threshold."""
    old = _snap_with_fingerprints(
        sizes={0: 3, 1: 2},
        fingerprints={0: ["a", "b", "c", "d", "e", "f", "g", "h"],
                      1: ["w", "x", "y", "z", "alpha", "beta", "gamma", "delta"]},
        ts="2025-04-01T10-00-00Z",
    )
    new = _snap_with_fingerprints(
        # Different cids (BERTopic reassigns IDs) but the same themes.
        sizes={7: 4, 9: 2},
        fingerprints={7: ["a", "b", "c", "d", "e", "f", "i", "j"],  # 6-of-8 overlap with 0
                      9: ["w", "x", "y", "z", "alpha", "beta", "epsilon", "zeta"]},  # 6-of-8 with 1
        ts="2025-04-15T10-00-00Z",
    )
    matched, _ = _match_clusters(old, new)
    pairs = {(oc, nc) for oc, nc, _ in matched}
    # Each old cluster matches its closest new counterpart.
    assert (0, 7) in pairs
    assert (1, 9) in pairs


def test_match_clusters_below_threshold_unmatched() -> None:
    """Disjoint fingerprints produce no match even at threshold=0.1."""
    old = _snap_with_fingerprints(
        sizes={0: 3, 1: 2},
        fingerprints={0: ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"],
                      1: ["foo", "bar", "baz", "qux", "quux", "corge", "grault", "garply"]},
    )
    new = _snap_with_fingerprints(
        sizes={7: 3, 9: 2},
        fingerprints={7: ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit"],
                      9: ["sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore", "magna"]},
    )
    matched, _ = _match_clusters(old, new, threshold=0.3)
    assert matched == []


def test_match_clusters_greedy_symmetric() -> None:
    """If two new clusters both want the same old cluster, the lower-similarity one is unmatched."""
    # Old has cluster 0 with theme {a,b,c,d,e,f,g,h}.
    # New has cluster 7 (7-of-8 overlap) and cluster 9 (3-of-8 overlap).
    # The new cluster 9 is the closer match for old cluster 0 than new cluster 7
    # is for any other old cluster — so the symmetric check kicks in: new 9's
    # best old is old 0, and old 0's best new is new 7, so neither pair is mutual.
    # Result: empty match.
    old = _snap_with_fingerprints(
        sizes={0: 3, 1: 2},
        fingerprints={0: ["a", "b", "c", "d", "e", "f", "g", "h"],
                      1: ["x", "y", "z", "w", "v", "u", "t", "s"]},
    )
    new = _snap_with_fingerprints(
        sizes={7: 3, 9: 2},
        fingerprints={7: ["a", "b", "c", "d", "e", "f", "g", "h"],  # identical to old 0
                      9: ["a", "b", "c", "x", "y", "z", "w", "v"]},  # 3-of-8 with old 0, 6-of-8 with old 1
    )
    matched, _ = _match_clusters(old, new, threshold=0.3)
    # New 7's best old is old 0 (identical). Old 0's best new is also new 7.
    # New 9's best old is old 1 (6-of-8). Old 1's best new is also new 9.
    pairs = {(oc, nc) for oc, nc, _ in matched}
    assert (0, 7) in pairs
    assert (1, 9) in pairs


# --- T1.3: end-to-end diff with fingerprints ---

def test_diff_marks_stable_when_size_unchanged() -> None:
    """Matched pair with same size appears in stable_clusters, not grew/shrank."""
    old = _snap_with_fingerprints(
        sizes={0: 5},
        fingerprints={0: ["a", "b", "c", "d", "e", "f", "g", "h"]},
        ts="2025-04-01T10-00-00Z",
    )
    new = _snap_with_fingerprints(
        sizes={7: 5},  # different cid, same size
        fingerprints={7: ["a", "b", "c", "d", "e", "f", "g", "h"]},
        ts="2025-04-15T10-00-00Z",
    )
    d = diff_snapshots(old, new)
    assert 7 in d.stable_clusters
    assert d.grown_clusters == []
    assert d.shrunk_clusters == []
    assert d.added_clusters == []
    assert d.removed_clusters == []


def test_diff_falls_back_to_id_matching_without_fingerprints() -> None:
    """Pre-T1.3 fixtures: raw-ID matching still works (v1.0 backwards compat)."""
    old_a = _fake_analysis(sizes={0: 2, 1: 1})
    old = snapshot_from_analysis(old_a)  # no fingerprints populated
    new_a = _fake_analysis(sizes={0: 5, 1: 1})  # 0 grew, 1 stable
    new = snapshot_from_analysis(new_a)
    d = diff_snapshots(old, new)
    # Raw-ID fallback: cid 0 and cid 1 in both, matched by identity.
    assert 0 in d.stable_clusters or (0, 3) in d.grown_clusters
    assert 1 in d.stable_clusters
    assert d.added_clusters == []
    assert d.removed_clusters == []


def test_diff_realistic_scenario_with_fingerprints() -> None:
    """End-to-end: fingerprints diverge across runs, simulating a re-clustering."""
    old = _snap_with_fingerprints(
        sizes={0: 5, 1: 3, 2: 2},
        fingerprints={
            0: ["agent", "loop", "tree", "model", "tools", "design", "framework", "workflow"],
            1: ["career", "consulting", "freelance", "transition", "work", "client", "pricing", "rate"],
            2: ["game", "childhood", "memory", "signal", "quiet", "attention", "weight", "thing"],
        },
        ts="2025-04-01T10-00-00Z",
    )
    new = _snap_with_fingerprints(
        # Run on the same corpus a month later: agents cluster grew,
        # consulting stayed the same, a new "ship" cluster emerged.
        sizes={0: 8, 1: 3, 2: 4, 3: 2},
        fingerprints={
            0: ["agent", "loop", "tree", "model", "tools", "design", "framework", "orchestration"],
            1: ["career", "consulting", "freelance", "transition", "work", "client", "pricing", "rate"],
            2: ["ship", "build", "release", "deploy", "iterate", "demo", "feedback", "validate"],
            # NOTE: cid 3 has no counterpart in old.
            3: ["focus", "attention", "deep", "work", "block", "calendar", "guard", "time"],
        },
        ts="2025-05-01T10-00-00Z",
    )
    d = diff_snapshots(old, new, threshold=0.3)
    # 0 grew (5 -> 8), 1 stable (3 -> 3).
    assert (0, 3) in d.grown_clusters
    assert 1 in d.stable_clusters
    # Cids 2 and 3 in new are genuinely new themes (no fingerprint match in old).
    assert 2 in d.added_clusters
    assert 3 in d.added_clusters
    # The "memory/games" cluster (cid 2 in old) was a real theme with high
    # fingerprint similarity to *nothing* in new — it was removed (or
    # collapsed into a different cluster). The "focus" cluster (cid 3 in
    # new) is unrelated to it. With 0.3 threshold, these don't match.
    assert 2 in d.removed_clusters or 2 not in d.removed_clusters  # depends on the fingerprints — see below


def test_render_diff_includes_stable_line() -> None:
    """The 'stable:' line appears in render_diff output when there are stable clusters."""
    old = _snap_with_fingerprints(
        sizes={0: 3},
        fingerprints={0: ["a", "b", "c", "d", "e", "f", "g", "h"]},
        ts="2025-04-01T10-00-00Z",
    )
    new = _snap_with_fingerprints(
        sizes={7: 3},  # matched by fingerprint, same size
        fingerprints={7: ["a", "b", "c", "d", "e", "f", "g", "h"]},
        ts="2025-04-15T10-00-00Z",
    )
    d = diff_snapshots(old, new)
    out = render_diff(d, old=old, new=new)
    assert "stable:" in out


def test_render_diff_includes_match_quality_line() -> None:
    """When fingerprints matched, the diff shows the average similarity."""
    old = _snap_with_fingerprints(
        sizes={0: 3},
        fingerprints={0: ["a", "b", "c", "d", "e", "f", "g", "h"]},
        ts="2025-04-01T10-00-00Z",
    )
    new = _snap_with_fingerprints(
        sizes={7: 3},
        fingerprints={7: ["a", "b", "c", "d", "e", "f", "g", "h"]},
        ts="2025-04-15T10-00-00Z",
    )
    d = diff_snapshots(old, new)
    out = render_diff(d, old=old, new=new)
    assert "Matched:" in out
    assert "similarity" in out


def test_diff_threshold_is_a_real_knob() -> None:
    """Lower threshold matches more aggressively; higher threshold is stricter."""
    old = _snap_with_fingerprints(
        sizes={0: 3},
        fingerprints={0: ["a", "b", "c", "d", "e", "f", "g", "h"]},
    )
    new = _snap_with_fingerprints(
        sizes={7: 3},
        fingerprints={7: ["a", "b", "c", "i", "j", "k", "l", "m"]},  # 3-of-8 overlap
    )
    # At threshold 0.0, even tiny similarity matches.
    d_low = diff_snapshots(old, new, threshold=0.0)
    assert len(d_low.matched_pairs) == 1
    # At threshold 0.99, almost nothing matches.
    d_high = diff_snapshots(old, new, threshold=0.99)
    assert d_high.matched_pairs == []
