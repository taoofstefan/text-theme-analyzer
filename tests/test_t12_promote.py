"""T1.2 tests: promote-to-project on stale verdicts.

Covers:
- `promote_key` synthesized in the LLM bundle.
- `_build_promote_keys` (JSON report) emits a top-level map.
- HTML dashboard renders a "Copy command" button only for
  `promote_to_project` verdicts; the button carries the right
  `data-promote-key` and `data-output-dir`.
- `render_promote_stub` produces a markdown stub with the heading,
  the `<!-- promote_key: ... -->` marker, the LLM reasoning, and
  representative notes.
- `apply_promotion` is idempotent: creates the file when missing,
  appends when no matching section, replaces in place when the
  key already exists. Honors `target_section`.
- `tta promote` CLI: looks up the key in `themes.json`, refuses
  non-`promote_to_project` verdicts, writes the target file.
- `Config.promote` and YAML round-trip for `target_file` and `sections`.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from click.testing import CliRunner

from text_theme_analyzer.config import (
    Config,
    apply_yaml_overrides,
)
from text_theme_analyzer.llm.enrichment import build_bundle
from text_theme_analyzer.llm.schemas import (
    ClusterAnnotation,
    EnrichmentResult,
    StaleVerdict,
)
from text_theme_analyzer.output.html_dashboard import render_html
from text_theme_analyzer.output.json_report import analysis_to_dict
from text_theme_analyzer.output.promote import (
    ClusterContext,
    apply_promotion,
    render_promote_stub,
)
from text_theme_analyzer.pipeline.model import (
    Analysis,
    ClusterResult,
    Note,
    NoteChunk,
    StaleIdea,
    ThemeTimeseries,
)

# --- helpers ---

def _make_note(nid: str, title: str = None, body: str = "body text", d: date = None,
               tags: list[str] | None = None) -> Note:
    return Note(
        id=nid,
        path=Path(f"notes/{nid}.md"),
        title=title or f"Title {nid}",
        body=body,
        date=d or date(2025, 1, 1),
        tags=tags or [],
    )


def _make_chunk(note_id: str, idx: int = 0, text: str = "some chunk text") -> NoteChunk:
    return NoteChunk(note_id=note_id, chunk_index=idx, text=text, char_offset=0)


def _fake_analysis_with_verdict(verdict_kind: str = "promote_to_project") -> Analysis:
    """Two-cluster, one-stale-idea analysis with a matching LLM verdict."""
    n1 = _make_note("n1", title="Anti-bullshit agency work",
                    body="Agency work anti-bullshit. Pricing reality check.",
                    d=date(2025, 4, 1), tags=["consulting"])
    n2 = _make_note("n2", title="Memory vault as life OS",
                    body="Vault as a life operating system. Daily notes feed it.",
                    d=date(2025, 3, 15), tags=["memory-vault"])
    chunks = [
        _make_chunk("n1", text=n1.body),
        _make_chunk("n2", text=n2.body),
    ]
    cluster_result = ClusterResult(
        assignments=[0, 1],
        cluster_sizes={0: 1, 1: 1},
        cluster_keywords={
            0: [("agency", 0.3), ("bullshit", 0.2), ("consulting", 0.1)],
            1: [("memory", 0.4), ("vault", 0.2), ("daily", 0.1)],
        },
        cluster_representatives={0: ["n1"], 1: ["n2"]},
        umap_2d=[(0.0, 0.0), (1.0, 1.0)],
        outlier_count=0,
    )
    timeseries = ThemeTimeseries(
        bucket="week",
        series={0: {date(2025, 4, 1): 1}, 1: {date(2025, 3, 15): 1}},
        spikes=[],
        stale=[
            StaleIdea(
                cluster_id=0, first_seen=date(2025, 1, 1), last_seen=date(2025, 4, 1),
                frequency=8, severity="strong", quiet_streak_buckets=12,
            ),
        ],
    )
    enrichment = EnrichmentResult(
        clusters=[
            ClusterAnnotation(
                cluster_id=0, name="Anti-bullshit agency",
                summary="A long-running tension between premium consulting and Fiverr reality.",
                top_quotes=["Agency work anti-bullshit."], emotional_tone="tense-curious",
            ),
            ClusterAnnotation(
                cluster_id=1, name="Memory vault OS",
                summary="The vault as a life operating system.", top_quotes=[], emotional_tone="calm",
            ),
        ],
        tensions=[],
        article_candidates=[],
        stale_recurring=[
            StaleVerdict(
                cluster_id=0, theme="Anti-bullshit agency work",
                verdict=verdict_kind,
                reasoning="The agency-vs-bullshit frame keeps coming back; worth promoting to a project plan.",
            ),
        ],
    )
    return Analysis(
        notes=[n1, n2],
        chunks=chunks,
        chunk_note_ids=["n1", "n2"],
        keywords={"n1": [("agency", 0.5)], "n2": [("memory", 0.5)]},
        keyphrase_frequency=[("agency", 1), ("memory", 1)],
        clusters=cluster_result,
        timeseries=timeseries,
        enrichment=enrichment,
        metadata={"date_range": ["2025-01-01", "2025-04-01"]},
    )


# --- promote_key in LLM bundle ---

def test_build_bundle_includes_promote_key() -> None:
    a = _fake_analysis_with_verdict()
    bundle = build_bundle(a)
    # Both clusters get a promote_key; cluster 0 is the one with the verdict.
    for c in bundle["clusters"]:
        assert "promote_key" in c
    # The promote_key for cluster 0 should be "0:2025-04-01" (last_seen.isoformat()).
    cid_0 = next(c for c in bundle["clusters"] if c["id"] == 0)
    assert cid_0["promote_key"] == "0:2025-04-01"
    cid_1 = next(c for c in bundle["clusters"] if c["id"] == 1)
    assert cid_1["promote_key"] == "1:2025-03-15"


# --- promote_keys in JSON report ---

def test_json_report_emits_promote_keys_map() -> None:
    a = _fake_analysis_with_verdict()
    d = analysis_to_dict(a)
    assert "promote_keys" in d
    assert "0:2025-04-01" in d["promote_keys"]
    rec = d["promote_keys"]["0:2025-04-01"]
    assert rec["cluster_id"] == 0
    assert rec["theme"] == "Anti-bullshit agency work"
    assert rec["verdict"] == "promote_to_project"
    assert "agency-vs-bullshit" in rec["reasoning"]
    assert rec["last_seen"] == "2025-04-01"
    assert rec["frequency"] == 8
    assert rec["severity"] == "strong"
    assert "agency" in rec["keywords"]
    assert rec["representative_note_ids"] == ["n1"]


def test_json_report_promote_keys_empty_when_no_enrichment() -> None:
    a = _fake_analysis_with_verdict()
    a.enrichment = None
    d = analysis_to_dict(a)
    assert d["promote_keys"] == {}


def test_json_report_promote_keys_empty_when_no_timeseries() -> None:
    a = _fake_analysis_with_verdict()
    a.timeseries = None
    d = analysis_to_dict(a)
    # No timeseries -> no cluster_id to join on -> empty map.
    assert d["promote_keys"] == {}


# --- HTML dashboard render ---

def test_html_dashboard_renders_promote_button_for_promote_verdict() -> None:
    a = _fake_analysis_with_verdict(verdict_kind="promote_to_project")
    html = render_html(a, output_dir=Path("/tmp/text-theme-output"))
    # The button appears, with the right key and the output_dir.
    assert "data-promote-key=\"0:2025-04-01\"" in html
    assert "Copy command" in html
    # The data-output-dir on the parent table carries the path.
    assert "data-output-dir=\"/tmp/text-theme-output\"" in html


def test_html_dashboard_omits_promote_button_for_non_promote_verdicts() -> None:
    for v in ("archive", "keep_observing"):
        a = _fake_analysis_with_verdict(verdict_kind=v)
        html = render_html(a, output_dir=Path("/tmp/x"))
        # No button for non-promote verdicts.
        assert "Copy command" not in html, f"button leaked for verdict={v}"


def test_html_dashboard_omits_promote_button_when_no_verdict() -> None:
    a = _fake_analysis_with_verdict(verdict_kind="promote_to_project")
    a.enrichment.stale_recurring = []  # No LLM verdicts at all.
    html = render_html(a, output_dir=Path("/tmp/x"))
    assert "Copy command" not in html


def test_html_dashboard_promote_key_matches_bundle_format() -> None:
    """The `data-promote-key` in the HTML equals the `promote_key` in the JSON map."""
    a = _fake_analysis_with_verdict()
    html = render_html(a, output_dir=Path("/tmp/x"))
    d = analysis_to_dict(a)
    for key in d["promote_keys"]:
        if d["promote_keys"][key]["verdict"] == "promote_to_project":
            assert f'data-promote-key="{key}"' in html


# --- render_promote_stub ---

def _ctx(**overrides) -> ClusterContext:
    base = dict(
        cluster_id=12,
        theme="Anti-bullshit agency work",
        reasoning="The agency-vs-bullshit frame keeps coming back; worth promoting.",
        last_seen="2025-04-01",
        first_seen="2024-11-15",
        frequency=8,
        severity="strong",
        keywords=["agency", "anti-bullshit", "premium consulting"],
        representative_notes=[
            ("n1", "2024-11-15", "First agency note", "notes/Self/2024-11-15 First.md"),
            ("n2", "2025-02-08", "Pricing reality", "notes/Themes/2025-02-08 Pricing.md"),
        ],
    )
    base.update(overrides)
    return ClusterContext(**base)


def test_render_promote_stub_has_required_pieces() -> None:
    stub = render_promote_stub("12:2025-04-01", _ctx())
    # Heading uses the verdict's theme (H3 — the file's `## ` is the bucket).
    assert "### Anti-bullshit agency work" in stub
    # Re-location marker: the invisible HTML comment with the key.
    assert "<!-- promote_key: 12:2025-04-01 -->" in stub
    # LLM reasoning shows up.
    assert "agency-vs-bullshit frame" in stub
    # Key phrases block.
    assert "agency, anti-bullshit, premium consulting" in stub
    # Standard markdown links (NOT wikilinks).
    assert "[First agency note](notes/Self/2024-11-15 First.md)" in stub
    assert "[Pricing reality](notes/Themes/2025-02-08 Pricing.md)" in stub
    # No wikilinks.
    assert "[[" not in stub and "]]" not in stub


def test_render_promote_stub_handles_no_keywords() -> None:
    stub = render_promote_stub("0:?", _ctx(keywords=[]))
    assert "Key phrases" not in stub
    assert "### " in stub  # heading still present


def test_render_promote_stub_handles_no_representative_notes() -> None:
    stub = render_promote_stub("0:?", _ctx(representative_notes=[]))
    assert "Supporting notes" not in stub


def test_render_promote_stub_normalizes_windows_backslashes() -> None:
    """Markdown link targets use forward slashes even on Windows."""
    stub = render_promote_stub("0:?", _ctx(representative_notes=[
        ("n1", "2025-01-01", "Title", "notes\\Self\\2025-01-01 Title.md"),
    ]))
    assert "[Title](notes/Self/2025-01-01 Title.md)" in stub


# --- apply_promotion (file-level) ---

def test_apply_promotion_creates_file_when_missing(tmp_path: Path) -> None:
    target = tmp_path / "promoted.md"
    stub = render_promote_stub("12:2025-04-01", _ctx())
    apply_promotion(target, stub, promote_key="12:2025-04-01")
    assert target.exists()
    content = target.read_text(encoding="utf-8")
    # Default bucket is "## Promoted"; stub lives under it.
    assert "## Promoted" in content
    # The stub's H3 project title is preserved.
    assert "### Anti-bullshit agency work" in content
    # Marker is in the body.
    assert "<!-- promote_key: 12:2025-04-01 -->" in content


def test_apply_promotion_appends_to_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "promoted.md"
    target.write_text("# Existing content\n\nSome preamble.\n", encoding="utf-8")
    stub = render_promote_stub("12:2025-04-01", _ctx())
    apply_promotion(target, stub, promote_key="12:2025-04-01")
    content = target.read_text(encoding="utf-8")
    # Preamble preserved.
    assert "Some preamble." in content
    # Stub appended under the default bucket.
    assert "## Promoted" in content
    assert "### Anti-bullshit agency work" in content
    # File ends with a newline.
    assert content.endswith("\n")


def test_apply_promotion_replaces_existing_section_by_promote_key(tmp_path: Path) -> None:
    target = tmp_path / "promoted.md"
    # First write.
    stub1 = render_promote_stub("12:2025-04-01", _ctx(reasoning="original reasoning."))
    apply_promotion(target, stub1, promote_key="12:2025-04-01")
    size_after_first = target.stat().st_size
    # Second write with new reasoning — should replace, not append.
    stub2 = render_promote_stub("12:2025-04-01", _ctx(reasoning="UPDATED reasoning."))
    apply_promotion(target, stub2, promote_key="12:2025-04-01")
    content = target.read_text(encoding="utf-8")
    assert "UPDATED reasoning." in content
    assert "original reasoning." not in content
    # File grew by a small amount (the size of the diff), not by a whole second section.
    assert target.stat().st_size < size_after_first + len(stub2)
    # Only one stub for this key.
    assert content.count("<!-- promote_key: 12:2025-04-01 -->") == 1
    # Only one `### ` heading for this theme.
    assert content.count("### Anti-bullshit agency work") == 1


def test_apply_promotion_with_sections_routes_to_named_heading(tmp_path: Path) -> None:
    target = tmp_path / "promoted.md"
    stub = render_promote_stub("12:2025-04-01", _ctx())
    apply_promotion(target, stub, promote_key="12:2025-04-01", target_section="In progress")
    content = target.read_text(encoding="utf-8")
    assert "## In progress" in content
    # The bucket heading comes before the stub's H3.
    assert content.index("## In progress") < content.index("### Anti-bullshit agency work")
    # The default "## Promoted" is NOT used.
    assert "## Promoted" not in content


def test_apply_promotion_preserves_existing_sections(tmp_path: Path) -> None:
    target = tmp_path / "promoted.md"
    # Write a different cluster first.
    stub_a = render_promote_stub("5:2025-01-01", _ctx(
        cluster_id=5, theme="Different project", reasoning="Other stuff.",
    ))
    apply_promotion(target, stub_a, promote_key="5:2025-01-01")
    # Now promote our test cluster.
    stub_b = render_promote_stub("12:2025-04-01", _ctx())
    apply_promotion(target, stub_b, promote_key="12:2025-04-01")
    content = target.read_text(encoding="utf-8")
    # Both stubs present, both under the same "## Promoted" bucket.
    assert "### Different project" in content
    assert "### Anti-bullshit agency work" in content
    # Re-locator markers for both.
    assert "<!-- promote_key: 5:2025-01-01 -->" in content
    assert "<!-- promote_key: 12:2025-04-01 -->" in content
    # Only one "## Promoted" bucket heading.
    assert content.count("## Promoted") == 1


def test_apply_promotion_three_stubs_in_one_bucket_idempotently(tmp_path: Path) -> None:
    """Re-promoting an existing key replaces; promoting a new key appends."""
    target = tmp_path / "promoted.md"
    a = render_promote_stub("1:2025-01-01", _ctx(cluster_id=1, theme="First"))
    b = render_promote_stub("2:2025-02-01", _ctx(cluster_id=2, theme="Second"))
    c_updated = render_promote_stub("1:2025-01-01", _ctx(cluster_id=1, theme="First", reasoning="UPDATED"))
    apply_promotion(target, a, promote_key="1:2025-01-01")
    apply_promotion(target, b, promote_key="2:2025-02-01")
    apply_promotion(target, c_updated, promote_key="1:2025-01-01")
    content = target.read_text(encoding="utf-8")
    # First stub is updated.
    assert "UPDATED" in content
    # Second stub is untouched.
    assert "### Second" in content
    # The original "First" reasoning is gone.
    assert content.count("<!-- promote_key: 1:2025-01-01 -->") == 1
    assert content.count("<!-- promote_key: 2:2025-02-01 -->") == 1
    # Both `### ` headings present, exactly once each.
    assert content.count("### First") == 1
    assert content.count("### Second") == 1


# --- config round-trip ---

def test_config_promote_defaults() -> None:
    cfg = Config()
    assert cfg.promote.target_file == Path("promoted-projects.md")
    assert cfg.promote.sections == []


def test_config_promote_yaml_override() -> None:
    cfg = Config()
    apply_yaml_overrides(cfg, {
        "promote": {
            "target_file": "kanban.md",
            "sections": ["To start", "In progress", "Archive"],
        }
    })
    assert cfg.promote.target_file == Path("kanban.md")
    assert cfg.promote.sections == ["To start", "In progress", "Archive"]


def test_config_promote_partial_yaml_override_preserves_unset() -> None:
    cfg = Config()
    apply_yaml_overrides(cfg, {"promote": {"target_file": "x.md"}})
    assert cfg.promote.target_file == Path("x.md")
    assert cfg.promote.sections == []  # default preserved


# --- CLI subcommand ---

def _write_themes_json(run_dir: Path, verdict_kind: str = "promote_to_project") -> None:
    """Helper: write a minimal themes.json to `run_dir` for CLI tests."""
    run_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": "1.0",
        "metadata": {},
        "summary": {"note_count": 2, "chunk_count": 2},
        "keyphrases": {"corpus_frequency": [], "per_note": {}},
        "clusters": {
            "count": 2, "outlier_count": 0, "cluster_ids": [0, 1],
            "sizes": [{"cluster_id": 0, "size": 1}, {"cluster_id": 1, "size": 1}],
            "keywords": {
                "0": [{"word": "agency", "score": 0.3}],
                "1": [{"word": "memory", "score": 0.3}],
            },
            "representatives": {"0": ["n1"], "1": ["n2"]},
            "umap_2d": [],
        },
        "timeseries": {
            "bucket": "week", "cluster_ids": [0, 1],
            "series": {"0": {"2025-04-01": 1}, "1": {"2025-03-15": 1}},
            "spikes": [],
            "stale": [
                {"cluster_id": 0, "first_seen": "2025-01-01", "last_seen": "2025-04-01",
                 "frequency": 8, "severity": "strong", "quiet_streak_buckets": 12},
            ],
        },
        "enrichment": {
            "clusters": [],
            "tensions": [],
            "article_candidates": [],
            "stale_recurring": [
                {"cluster_id": 0, "theme": "Anti-bullshit agency work",
                 "verdict": verdict_kind, "reasoning": "Test reasoning text."},
            ],
            "quote_validation": {"requested": 0, "kept": 0, "dropped": 0, "examples": []},
        },
        "promote_keys": {
            "0:2025-04-01": {
                "cluster_id": 0, "theme": "Anti-bullshit agency work",
                "verdict": verdict_kind, "reasoning": "Test reasoning text.",
                "last_seen": "2025-04-01", "first_seen": "2025-01-01",
                "frequency": 8, "severity": "strong",
                "keywords": ["agency", "anti-bullshit"],
                "representative_note_ids": ["n1"],
            }
        },
        "files": [
            {"id": "n1", "path": "notes/Self/2025-04-01 Anti-bullshit agency work.md",
             "title": "Anti-bullshit agency work", "date": "2025-04-01",
             "tags": ["consulting"], "word_count": 50},
            {"id": "n2", "path": "notes/Themes/2025-03-15 Memory vault.md",
             "title": "Memory vault", "date": "2025-03-15", "tags": [], "word_count": 30},
        ],
    }
    (run_dir / "themes.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_cli_promote_writes_target_file(tmp_path: Path) -> None:
    from text_theme_analyzer.cli import main
    runner = CliRunner()
    run_dir = tmp_path / "text-theme-output" / "2026-06-10T00-00-00Z"
    _write_themes_json(run_dir)
    target = tmp_path / "promoted.md"
    result = runner.invoke(
        main,
        ["promote", "0:2025-04-01", "--from-run", str(run_dir), "--target-file", str(target)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert target.exists()
    content = target.read_text(encoding="utf-8")
    # Default bucket heading is present.
    assert "## Promoted" in content
    # Stub H3 (project title) is present.
    assert "### Anti-bullshit agency work" in content
    # Marker is in the body.
    assert "<!-- promote_key: 0:2025-04-01 -->" in content
    # LLM reasoning shows up in the body.
    assert "Test reasoning text." in content
    # Source-note link uses standard markdown (not wikilinks).
    assert "[Anti-bullshit agency work](notes/Self/2025-04-01 Anti-bullshit agency work.md)" in content


def test_cli_promote_unknown_key_errors_cleanly(tmp_path: Path) -> None:
    from text_theme_analyzer.cli import main
    runner = CliRunner()
    run_dir = tmp_path / "text-theme-output" / "run1"
    _write_themes_json(run_dir)
    target = tmp_path / "promoted.md"
    result = runner.invoke(
        main,
        ["promote", "99:2099-01-01", "--from-run", str(run_dir), "--target-file", str(target)],
    )
    assert result.exit_code != 0
    assert "99:2099-01-01" in result.output
    # No file should have been created.
    assert not target.exists()


def test_cli_promote_archive_verdict_refuses(tmp_path: Path) -> None:
    from text_theme_analyzer.cli import main
    runner = CliRunner()
    run_dir = tmp_path / "text-theme-output" / "run1"
    _write_themes_json(run_dir, verdict_kind="archive")
    target = tmp_path / "promoted.md"
    result = runner.invoke(
        main,
        ["promote", "0:2025-04-01", "--from-run", str(run_dir), "--target-file", str(target)],
    )
    assert result.exit_code != 0
    assert "archive" in result.output.lower() or "not 'promote_to_project'" in result.output
    assert not target.exists()


def test_cli_promote_uses_latest_run_when_no_from_run(tmp_path: Path) -> None:
    from text_theme_analyzer.cli import main
    runner = CliRunner()
    out_dir = tmp_path / "text-theme-output"
    # Two runs; the second is newer. Set mtimes explicitly.
    older = out_dir / "older"
    newer = out_dir / "newer"
    _write_themes_json(older, verdict_kind="archive")  # no promote here
    _write_themes_json(newer, verdict_kind="promote_to_project")
    import os
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))
    target = tmp_path / "promoted.md"
    result = runner.invoke(
        main,
        ["promote", "0:2025-04-01", "--output-dir", str(out_dir), "--target-file", str(target)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert target.exists()


def test_cli_promote_missing_themes_json_errors(tmp_path: Path) -> None:
    from text_theme_analyzer.cli import main
    runner = CliRunner()
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    result = runner.invoke(
        main,
        ["promote", "0:2025-04-01", "--from-run", str(run_dir)],
    )
    assert result.exit_code != 0
    assert "themes.json" in result.output
