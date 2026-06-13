"""T2.2 tests: `tta run-all` per-corpus orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from text_theme_analyzer.cli import main
from text_theme_analyzer.config import (
    Config,
    CorpusConfig,
    OutputFormat,
    _apply_corpus_overrides,
    apply_yaml_overrides,
)
from text_theme_analyzer.output.run_all import _render_index, run_all

# --- config plumbing ---

def test_apply_corpus_overrides_inherits_globals() -> None:
    base = Config()
    base.input_path = Path("/global/in")
    base.output_dir = Path("/global/out")
    base.tag_weight = 0.5
    corpus = CorpusConfig()
    cfg = _apply_corpus_overrides(base, "self", corpus)
    assert cfg.input_path == Path("/global/in")
    assert cfg.output_dir == Path("/global/out/self")
    assert cfg.tag_weight == 0.5
    assert cfg.outputs == [OutputFormat.CLI]


def test_apply_corpus_overrides_respects_corpus_values() -> None:
    base = Config()
    base.output_dir = Path("/global/out")
    corpus = CorpusConfig(
        input_path=Path("/self"),
        output_dir=Path("/self-out"),
        tag_weight=0.9,
        include=["**/*.md"],
    )
    cfg = _apply_corpus_overrides(base, "self", corpus)
    assert cfg.input_path == Path("/self")
    assert cfg.output_dir == Path("/self-out")
    assert cfg.tag_weight == 0.9
    assert cfg.include == ["**/*.md"]


def test_yaml_corpora_block_parses() -> None:
    cfg = Config()
    data = yaml.safe_load("""
corpora:
  self:
    input_path: ./Self
    tag_weight: 0.3
  reading:
    input_path: ./06 Reading
    output_dir: ./out/reading
""")
    apply_yaml_overrides(cfg, data)
    assert "self" in cfg.corpora
    assert "reading" in cfg.corpora
    assert cfg.corpora["self"].input_path == Path("./Self")
    assert cfg.corpora["self"].tag_weight == 0.3
    assert cfg.corpora["reading"].output_dir == Path("./out/reading")


# --- index rendering ---

def test_render_index_links_to_dashboards() -> None:
    html = _render_index([
        {"name": "self", "output_dir": "out/self", "notes": 50, "clusters": 7, "rel_html": "self/dashboard.html"},
        {"name": "reading", "output_dir": "out/reading", "notes": 12, "clusters": 2, "rel_html": "reading/dashboard.html"},
    ])
    assert 'href="self/dashboard.html"' in html
    assert 'href="reading/dashboard.html"' in html
    assert "self" in html
    assert "50 notes" in html


# --- run_all integration ---

@pytest.fixture
def two_corpus_config(tmp_path: Path) -> Config:
    base = tmp_path / "vault"
    self_dir = base / "Self"
    reading_dir = base / "Reading"
    self_dir.mkdir(parents=True)
    reading_dir.mkdir(parents=True)
    (self_dir / "2025-01-01-a.md").write_text("# A\n\nagent workflow design", encoding="utf-8")
    (self_dir / "2025-01-02-b.md").write_text("# B\n\nagent design tooling", encoding="utf-8")
    (reading_dir / "2025-01-01-c.md").write_text("# C\n\ncrypto market analysis", encoding="utf-8")

    cfg = Config()
    cfg.input_path = base
    cfg.output_dir = tmp_path / "out"
    cfg.no_llm = True
    cfg.no_cache = True
    cfg.outputs = [OutputFormat.HTML, OutputFormat.JSON]
    cfg.corpora = {
        "self": CorpusConfig(input_path=self_dir),
        "reading": CorpusConfig(input_path=reading_dir),
    }
    return cfg


def test_run_all_writes_outputs_and_index(two_corpus_config: Config) -> None:
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("bertopic")
    pytest.importorskip("umap")
    pytest.importorskip("hdbscan")
    results, index_path = run_all(two_corpus_config, write_history=False)
    assert len(results) == 2
    names = {n for n, _, _ in results}
    assert names == {"self", "reading"}
    assert index_path.is_file()
    assert (two_corpus_config.output_dir / "self" / "dashboard.html").is_file()
    assert (two_corpus_config.output_dir / "reading" / "dashboard.html").is_file()


def test_run_all_cli_wires_through_config(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    self_dir = vault / "Self"
    self_dir.mkdir(parents=True)
    (self_dir / "2025-01-01-a.md").write_text("# A\n\nagent workflow design", encoding="utf-8")

    config_yml = tmp_path / "text-theme-analyzer.yml"
    config_yml.write_text(
        "corpora:\n  self:\n    input_path: " + str(self_dir.as_posix()) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run-all",
            "--config", str(config_yml),
            "--output-dir", str(out_dir),
            "--no-llm",
            "--no-cache",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "self" / "themes.json").is_file()
    assert (out_dir / "corpora" / "index.html").is_file()
