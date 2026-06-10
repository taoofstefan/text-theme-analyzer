"""Pins the config-precedence contract: CLI > env > YAML > defaults.

A regression of this ordering caused the M5 e2e test to silently write
artifacts to the wrong location when a `text-theme-analyzer.yml` was
present in the CWD. These tests exercise the layering directly so a
future refactor can't reintroduce the bug without breaking the suite.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from text_theme_analyzer.cli import main
from text_theme_analyzer.config import (
    Config,
    apply_env_overrides,
    apply_yaml_overrides,
    load_yaml_config,
)

# --- helpers ---


def _write_two_note_corpus(notes_dir: Path) -> None:
    notes_dir.mkdir(parents=True, exist_ok=True)
    (notes_dir / "a.md").write_text(
        "---\ndate: 2025-04-01\ntitle: A\ntags: [test]\n---\n\nSome text about agents and design.\n",
        encoding="utf-8",
    )
    (notes_dir / "b.md").write_text(
        "---\ndate: 2025-04-15\ntitle: B\ntags: [test]\n---\n\nMore text about agent design and workflow.\n",
        encoding="utf-8",
    )


def _run_in(working_dir: Path, args: list[str]) -> Path:
    """Run the CLI from `working_dir` (so find_config_file looks there)
    and return the directory that received themes-report.md. Raises
    AssertionError if the CLI exits non-zero or no report was written.
    """
    old_cwd = Path.cwd()
    try:
        os.chdir(working_dir)
        runner = CliRunner()
        result = runner.invoke(main, ["analyze", *args, "--no-llm", "-q"], catch_exceptions=False)
    finally:
        os.chdir(old_cwd)
    assert result.exit_code == 0, f"CLI failed:\n{result.output}"
    for line in result.output.splitlines():
        if "themes-report.md" in line:
            # Format: "wrote <path>\themes-report.md". Resolve relative to working_dir.
            rel = line.split("wrote", 1)[1].strip()
            return (working_dir / Path(rel).parent).resolve()
    raise AssertionError(f"Could not find themes-report.md in CLI output:\n{result.output}")


# --- tests ---


def test_yaml_overrides_default_output_dir(tmp_path: Path) -> None:
    """No CLI --output-dir: the YAML's output_dir must beat the dataclass default."""
    pytest.importorskip("yake")
    notes = tmp_path / "notes"
    _write_two_note_corpus(notes)
    yaml_dir = tmp_path / "from_yaml"
    (tmp_path / "text-theme-analyzer.yml").write_text(
        f"output_dir: {yaml_dir.as_posix()}\noutputs: [markdown]\n",
        encoding="utf-8",
    )

    out = _run_in(tmp_path, [str(notes)])
    assert out == yaml_dir.resolve(), f"YAML output_dir not honored; got {out}"


def test_cli_flag_overrides_yaml_output_dir(tmp_path: Path) -> None:
    """CLI --output-dir must beat the YAML's output_dir (the original bug)."""
    pytest.importorskip("yake")
    notes = tmp_path / "notes"
    _write_two_note_corpus(notes)
    yaml_dir = tmp_path / "from_yaml"
    cli_dir = tmp_path / "from_cli"
    (tmp_path / "text-theme-analyzer.yml").write_text(
        f"output_dir: {yaml_dir.as_posix()}\noutputs: [markdown]\n",
        encoding="utf-8",
    )

    out = _run_in(tmp_path, [str(notes), "--output-dir", str(cli_dir)])
    assert out == cli_dir.resolve(), f"CLI --output-dir not honored over YAML; got {out}"


def test_env_var_overrides_yaml_cache_dir(tmp_path: Path, monkeypatch) -> None:
    """Env vars read by apply_env_overrides must beat the YAML's value.

    `cache_dir` is the only field currently read from the env. We pin the
    layering at the config-construction level rather than the output level
    so this stays a fast, deterministic test.
    """
    yaml_cache = tmp_path / "yaml_cache"
    env_cache = tmp_path / "env_cache"
    yaml_path = tmp_path / "text-theme-analyzer.yml"
    yaml_path.write_text(f"cache_dir: {yaml_cache.as_posix()}\n", encoding="utf-8")

    monkeypatch.setenv("TEXTHEME_CACHE_DIR", str(env_cache))

    cfg = Config()
    apply_yaml_overrides(cfg, load_yaml_config(yaml_path))
    apply_env_overrides(cfg)

    assert cfg.cache_dir == Path(env_cache), f"env var did not win; cache_dir={cfg.cache_dir}"
