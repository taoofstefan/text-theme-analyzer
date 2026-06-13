"""T1.2a tests: LLM-picked `target_section` for promote stubs."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from text_theme_analyzer.config import Config, apply_yaml_overrides
from text_theme_analyzer.llm.prompts import build_user_prompt
from text_theme_analyzer.llm.schemas import EnrichmentResult, StaleVerdict
from text_theme_analyzer.output.json_report import _build_promote_keys
from text_theme_analyzer.output.promote import _select_bucket_heading
from text_theme_analyzer.pipeline.model import (
    Analysis,
    ClusterResult,
    Note,
    ThemeTimeseries,
)

# --- _select_bucket_heading ---

def test_select_bucket_uses_target_section_when_configured() -> None:
    assert _select_bucket_heading([], "Archive", ["To start", "In progress", "Archive"]) == "## Archive"


def test_select_bucket_falls_back_to_first_configured_section() -> None:
    assert _select_bucket_heading([], None, ["To start", "Archive"]) == "## To start"


def test_select_bucket_falls_back_to_promoted() -> None:
    assert _select_bucket_heading([], None, []) == "## Promoted"


def test_select_bucket_ignores_unknown_target_section() -> None:
    """An LLM-named section not in the config falls back to the first configured section."""
    assert _select_bucket_heading([], "Not configured", ["To start", "Archive"]) == "## To start"


# --- prompt includes sections ---

def test_prompt_includes_section_hint_when_sections_given() -> None:
    user = build_user_prompt(
        total_notes=1,
        date_range=("2025-01-01", "2025-01-01"),
        clusters=[],
        spikes=[],
        stale_candidates=[],
        promote_sections=["To start", "In progress", "Archive"],
    )
    assert "To start" in user
    assert "target_section" in user
    assert "promote_to_project" in user


def test_prompt_omits_section_hint_when_no_sections() -> None:
    user = build_user_prompt(
        total_notes=1,
        date_range=("2025-01-01", "2025-01-01"),
        clusters=[],
        spikes=[],
        stale_candidates=[],
    )
    # The schema still names the optional field, but no section list/nudge appears.
    assert "user's configured project-board sections" not in user
    assert "To start" not in user


# --- schema round-trip ---

def test_stale_verdict_accepts_target_section() -> None:
    result = EnrichmentResult.model_validate({
        "clusters": [],
        "tensions": [],
        "article_candidates": [],
        "stale_recurring": [
            {
                "cluster_id": 1,
                "theme": "foo",
                "verdict": "promote_to_project",
                "reasoning": "bar baz qux",
                "target_section": "Archive",
            }
        ],
    })
    assert len(result.stale_recurring) == 1
    verdict = result.stale_recurring[0]
    assert isinstance(verdict, StaleVerdict)
    assert verdict.target_section == "Archive"


# --- JSON report surfaces target_section ---

def _fake_analysis_with_stale(*, target_section: str | None = None) -> Analysis:
    n1 = Note(id="n1", path=Path("n1.md"), title="A", body="body", date=date(2025, 1, 1))
    cluster = ClusterResult(
        assignments=[0],
        cluster_sizes={0: 1},
        cluster_keywords={0: [("kw", 0.5)]},
        cluster_representatives={0: ["n1"]},
        umap_2d=[(0.0, 0.0)],
        outlier_count=0,
    )
    ts = ThemeTimeseries(
        bucket="week",
        series={0: {}},
        stale=[
            type("StaleIdea", (), {
                "cluster_id": 0,
                "first_seen": date(2025, 1, 1),
                "last_seen": date(2025, 1, 1),
                "frequency": 3,
                "severity": "medium",
                "quiet_streak_buckets": 0,
            })()
        ],
    )
    enrichment = EnrichmentResult.model_validate({
        "clusters": [],
        "tensions": [],
        "article_candidates": [],
        "stale_recurring": [
            {
                "cluster_id": 0,
                "theme": "A theme",
                "verdict": "promote_to_project",
                "reasoning": "Because it matters.",
                "target_section": target_section,
            }
        ],
    })
    return Analysis(
        notes=[n1],
        chunks=[],
        chunk_note_ids=[],
        keywords={},
        keyphrase_frequency=[],
        clusters=cluster,
        timeseries=ts,
        enrichment=enrichment,
        metadata={"config": {"promote_sections": ["To start", "Archive"]}},
    )


def test_json_report_includes_target_section() -> None:
    analysis = _fake_analysis_with_stale(target_section="Archive")
    keys = _build_promote_keys(analysis)
    assert "0:2025-01-01" in keys
    assert keys["0:2025-01-01"]["target_section"] == "Archive"


def test_json_report_null_target_section_when_unset() -> None:
    analysis = _fake_analysis_with_stale(target_section=None)
    keys = _build_promote_keys(analysis)
    assert keys["0:2025-01-01"]["target_section"] is None


# --- config plumbing for promote sections ---

def test_config_promote_sections_yaml_override() -> None:
    cfg = Config()
    apply_yaml_overrides(cfg, {"promote": {"sections": ["To start", "In progress", "Archive"]}})
    assert cfg.promote.sections == ["To start", "In progress", "Archive"]
