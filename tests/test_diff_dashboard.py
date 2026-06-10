"""T1.3 tests: 2-column diff HTML dashboard + CLI wiring."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from text_theme_analyzer.cli import main
from text_theme_analyzer.output.diff_dashboard import render_diff_html
from text_theme_analyzer.output.history import (
    diff_snapshots,
    write_snapshot,
)


def _snap(
    *,
    sizes: dict[int, int],
    fingerprints: dict[int, list[str]],
    ts: str = "2025-04-01T10-00-00Z",
) -> RunSnapshot:  # noqa: F821
    from text_theme_analyzer.output.history import RunSnapshot
    return RunSnapshot(
        timestamp=ts,
        note_count=20,
        chunk_count=20,
        date_range=["2025-04-01", "2025-04-15"],
        keyphrase_top=[("agent", 5), ("model", 5)],
        cluster_sizes=sizes,
        cluster_keywords={cid: fps[:5] for cid, fps in fingerprints.items()},
        cluster_fingerprints=fingerprints,
        cluster_names={},
        spike_count=0,
        stale_count=0,
    )


# --- render_diff_html ---

def test_render_diff_html_is_self_contained() -> None:
    """The HTML has no external resource references (no http://, no <script src=...)."""
    old = _snap(
        sizes={0: 3},
        fingerprints={0: ["a", "b", "c", "d", "e", "f", "g", "h"]},
    )
    new = _snap(
        sizes={7: 5},
        fingerprints={7: ["a", "b", "c", "d", "e", "f", "g", "h"]},
        ts="2025-04-15T10-00-00Z",
    )
    d = diff_snapshots(old, new)
    html = render_diff_html(d, old=old, new=new)
    assert "<!doctype html>" in html.lower()
    assert "<html" in html
    # No external network requests.
    assert "http://" not in html
    assert "https://" not in html
    assert "<script src=" not in html
    # No external CSS link.
    assert "<link" not in html


def test_render_diff_html_includes_timestamps() -> None:
    old = _snap(
        sizes={0: 3},
        fingerprints={0: ["a", "b", "c", "d", "e", "f", "g", "h"]},
    )
    new = _snap(
        sizes={7: 5},
        fingerprints={7: ["a", "b", "c", "d", "e", "f", "g", "h"]},
        ts="2025-04-15T10-00-00Z",
    )
    d = diff_snapshots(old, new)
    html = render_diff_html(d, old=old, new=new)
    assert "2025-04-01T10-00-00Z" in html
    assert "2025-04-15T10-00-00Z" in html


def test_render_diff_html_renders_matched_table() -> None:
    """A matched pair shows up in the matched-clusters table with similarity."""
    old = _snap(
        sizes={0: 3},
        fingerprints={0: ["agent", "loop", "tree", "model", "tools",
                          "design", "framework", "workflow"]},
    )
    new = _snap(
        sizes={7: 5},
        fingerprints={7: ["agent", "loop", "tree", "model", "tools",
                          "design", "framework", "workflow"]},
        ts="2025-04-15T10-00-00Z",
    )
    d = diff_snapshots(old, new)
    html = render_diff_html(d, old=old, new=new)
    assert "Matched clusters" in html
    # The pair shows similarity score (1.00 for identical).
    assert "1.00" in html
    # Both cluster ids appear in the table.
    assert "#0" in html
    assert "#7" in html
    # Verdict pill for the stable pair.
    assert "stable" in html


def test_render_diff_html_renders_added_section() -> None:
    """An added cluster (no fingerprint match) appears in the 'added' section."""
    old = _snap(
        sizes={0: 3},
        fingerprints={0: ["alpha", "beta", "gamma", "delta",
                          "epsilon", "zeta", "eta", "theta"]},
    )
    new = _snap(
        sizes={0: 3, 7: 2},  # cluster 7 has no counterpart
        fingerprints={
            0: ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"],
            7: ["ship", "build", "release", "deploy", "iterate", "demo", "feedback", "validate"],
        },
        ts="2025-04-15T10-00-00Z",
    )
    d = diff_snapshots(old, new)
    html = render_diff_html(d, old=old, new=new)
    assert "Added clusters" in html
    assert "Removed clusters" not in html  # nothing was removed
    # The new cluster's keyword fingerprint shows in the added table.
    assert "ship" in html


def test_render_diff_html_renders_removed_section() -> None:
    """A removed cluster (no fingerprint match in new) appears in the 'removed' section."""
    old = _snap(
        sizes={0: 3, 1: 2},  # cluster 1 will be removed
        fingerprints={
            0: ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"],
            1: ["legacy", "scrap", "old", "thing", "stuff", "bit", "kind", "way"],
        },
    )
    new = _snap(
        sizes={0: 3},  # cluster 1 is gone
        fingerprints={0: ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]},
        ts="2025-04-15T10-00-00Z",
    )
    d = diff_snapshots(old, new)
    html = render_diff_html(d, old=old, new=new)
    assert "Removed clusters" in html
    assert "Added clusters" not in html
    assert "legacy" in html


def test_render_diff_html_renders_keyphrase_diff() -> None:
    """New and dropped keyphrases appear in the keyphrase-diff section."""
    old = _snap(
        sizes={0: 3},
        fingerprints={0: ["a", "b", "c", "d", "e", "f", "g", "h"]},
        ts="2025-04-01T10-00-00Z",
    )
    # Need to override keyphrase_top to add/remove phrases.
    old.keyphrase_top = [("agent", 5), ("legacy_phrase", 3)]
    new = old.__class__(
        timestamp="2025-04-15T10-00-00Z",
        note_count=20, chunk_count=20, date_range=None,
        keyphrase_top=[("agent", 5), ("tools", 4), ("new_phrase", 6)],
        cluster_sizes={7: 3},
        cluster_keywords={7: ["a", "b", "c", "d", "e"]},
        cluster_fingerprints={7: ["a", "b", "c", "d", "e", "f", "g", "h"]},
        cluster_names={}, spike_count=0, stale_count=0,
    )
    d = diff_snapshots(old, new)
    html = render_diff_html(d, old=old, new=new)
    assert "Keyphrase diff" in html
    assert "new_phrase" in html  # added
    assert "tools" in html       # added
    assert "legacy_phrase" in html  # dropped


def test_render_diff_html_renders_summary_block() -> None:
    """The summary block shows notes/chunks/clusters counts and spike/stale deltas."""
    old = _snap(
        sizes={0: 3},
        fingerprints={0: ["a", "b", "c", "d", "e", "f", "g", "h"]},
    )
    new = _snap(
        sizes={7: 5},
        fingerprints={7: ["a", "b", "c", "d", "e", "f", "g", "h"]},
        ts="2025-04-15T10-00-00Z",
    )
    d = diff_snapshots(old, new)
    html = render_diff_html(d, old=old, new=new)
    assert "Summary" in html
    assert "20 → 20" in html  # notes unchanged


def test_render_diff_html_escapes_xss_in_cluster_names() -> None:
    """Cluster labels are HTML-escaped (no <script> injection)."""
    old = _snap(
        sizes={0: 3},
        fingerprints={0: ["a", "b", "c", "d", "e", "f", "g", "h"]},
    )
    # Inject HTML into the cluster name. The Jinja autoescape must catch it.
    old.cluster_names[0] = "<script>alert('xss')</script>"
    new = _snap(
        sizes={7: 3},
        fingerprints={7: ["a", "b", "c", "d", "e", "f", "g", "h"]},
        ts="2025-04-15T10-00-00Z",
    )
    d = diff_snapshots(old, new)
    html = render_diff_html(d, old=old, new=new)
    # The literal <script> tag should NOT appear in the rendered output.
    assert "<script>alert" not in html
    # The escaped form is OK (and expected).
    assert "&lt;script&gt;" in html or "alert" not in html


# --- CLI integration ---

@pytest.fixture
def snapshots_dir(tmp_path: Path) -> Path:
    """A run-history/ dir with two snapshots that have fingerprints.

    Returns `tmp_path`, the parent of the run-history/ directory. The CLI
    expects `--output-dir` to point at this parent (it appends
    `run-history/` internally). The snapshots are written via
    `write_snapshot(snap, tmp_path)` which puts them at
    `tmp_path/run-history/{ts}.json`.
    """
    old = _snap(
        sizes={0: 3, 1: 2},
        fingerprints={0: ["agent", "loop", "tree", "model", "tools",
                          "design", "framework", "workflow"],
                      1: ["career", "consulting", "freelance", "transition",
                          "work", "client", "pricing", "rate"]},
    )
    new = _snap(
        sizes={7: 5, 8: 2},  # different cids, same themes + sizes
        fingerprints={7: ["agent", "loop", "tree", "model", "tools",
                          "design", "framework", "orchestration"],
                      8: ["career", "consulting", "freelance", "transition",
                          "work", "client", "pricing", "rate"]},
        ts="2025-04-15T10-00-00Z",
    )
    write_snapshot(old, tmp_path)
    write_snapshot(new, tmp_path)
    return tmp_path


def test_cli_diff_writes_html(snapshots_dir: Path) -> None:
    """`tta diff OLD NEW --html out.html` writes the HTML file and prints the text diff."""
    out_html = snapshots_dir / "diff.html"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "diff", "2025-04-01T10-00-00Z", "2025-04-15T10-00-00Z",
            "--output-dir", str(snapshots_dir),
            "--html", str(out_html),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert out_html.is_file()
    # Text output is on stdout.
    assert "Run diff:" in result.output
    # HTML file is well-formed.
    html = out_html.read_text(encoding="utf-8")
    assert "<!doctype html>" in html.lower()
    # The HTML file is mentioned in stderr (echoed by the CLI).
    assert "wrote" in (result.stderr or "")


def test_cli_diff_without_html_does_not_write_file(snapshots_dir: Path) -> None:
    """Without --html, only the text diff is printed and no file is written."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "diff", "2025-04-01T10-00-00Z", "2025-04-15T10-00-00Z",
            "--output-dir", str(snapshots_dir),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    # No HTML file created.
    html_files = list(snapshots_dir.glob("*.html"))
    assert html_files == []
    # Text output present.
    assert "Run diff:" in result.output


def test_cli_diff_unknown_snapshot_errors(snapshots_dir: Path) -> None:
    """Unknown timestamp produces a clear error."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "diff", "2099-01-01T00-00-00Z", "2025-04-15T10-00-00Z",
            "--output-dir", str(snapshots_dir),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code != 0
    # The error mentions the missing snapshot.
    assert "not found" in (result.output or "")


def test_cli_diff_help_lists_new_flags() -> None:
    """`tta diff --help` shows the --html and --match-threshold flags."""
    runner = CliRunner()
    result = runner.invoke(main, ["diff", "--help"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "--html" in result.output
    assert "--match-threshold" in result.output
