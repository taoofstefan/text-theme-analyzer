"""Run history + diff tests."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from text_theme_analyzer.output.history import (
    HISTORY_DIRNAME,
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
    cluster_kws = {cid: [(f"kw{cid}", 0.3), ("agent", 0.2)] for cid in cluster_sizes}
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
