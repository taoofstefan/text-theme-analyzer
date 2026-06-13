"""T2.1 tests: --require-dates raises on notes without explicit dates."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from text_theme_analyzer.cli import main
from text_theme_analyzer.config import Config, apply_yaml_overrides
from text_theme_analyzer.pipeline.orchestrator import run
from text_theme_analyzer.utils.dates import has_authoritative_date

# --- has_authoritative_date ---

def test_authoritative_date_frontmatter() -> None:
    assert has_authoritative_date({"date": "2025-01-01"}, Path("note.md"))


def test_authoritative_date_filename() -> None:
    assert has_authoritative_date({}, Path("2025-01-01-note.md"))


def test_authoritative_date_rejects_mtime_only() -> None:
    assert not has_authoritative_date({}, Path("note.md"))


def test_authoritative_date_rejects_empty_frontmatter() -> None:
    assert not has_authoritative_date({"date": ""}, Path("note.md"))


# --- config plumbing ---

def test_config_require_dates_default_false() -> None:
    cfg = Config()
    assert cfg.require_dates is False


def test_config_require_dates_yaml_override() -> None:
    cfg = Config()
    apply_yaml_overrides(cfg, {"require_dates": True})
    assert cfg.require_dates is True


# --- orchestrator behavior ---

def _config_for(tmp_path: Path, *, require_dates: bool) -> Config:
    cfg = Config()
    cfg.input_path = tmp_path / "notes"
    cfg.require_dates = require_dates
    cfg.no_llm = True
    cfg.no_cache = True
    return cfg


def test_orchestrator_allows_undated_when_not_required(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbody", encoding="utf-8")
    cfg = _config_for(tmp_path, require_dates=False)
    # Should complete without raising.
    analysis = run(cfg)
    assert len(analysis.notes) == 1


def test_orchestrator_raises_on_undated_when_required(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbody", encoding="utf-8")
    cfg = _config_for(tmp_path, require_dates=True)
    with pytest.raises(ValueError, match="--require-dates"):
        run(cfg)


def test_orchestrator_allows_dated_notes_when_required(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text(
        "---\ndate: 2025-01-01\n---\n\n# A\n\nbody",
        encoding="utf-8",
    )
    (notes / "2025-02-01-b.md").write_text("# B\n\nbody", encoding="utf-8")
    cfg = _config_for(tmp_path, require_dates=True)
    analysis = run(cfg)
    assert len(analysis.notes) == 2


def test_orchestrator_error_lists_all_undated_paths(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbody", encoding="utf-8")
    (notes / "b.md").write_text("# B\n\nbody", encoding="utf-8")
    cfg = _config_for(tmp_path, require_dates=True)
    with pytest.raises(ValueError, match=r"a\.md") as exc_info:
        run(cfg)
    assert "b.md" in str(exc_info.value)


# --- CLI wiring ---

def test_cli_require_dates_flag(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("# A\n\nbody", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["analyze", str(notes), "--no-llm", "--no-cache", "--require-dates", "--output-dir", str(tmp_path / "out")],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    exc_text = str(result.exception)
    assert "no explicit date" in exc_text
    assert "a.md" in exc_text
