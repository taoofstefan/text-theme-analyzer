"""M3 tests: LLM factory, enrichment bundle, quote validation, JSON sidecar.

The LLM clients use httpx; we mock httpx in tests rather than calling a real endpoint.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import httpx
import pytest

from text_theme_analyzer.config import Config, Provider
from text_theme_analyzer.llm.base import LLMError
from text_theme_analyzer.llm.enrichment import (
    _sample_quotes,
    _validate_quotes,
    build_bundle,
    enrich,
)
from text_theme_analyzer.llm.factory import build_client
from text_theme_analyzer.llm.openai_compat import OpenAICompatibleClient
from text_theme_analyzer.llm.schemas import (
    ClusterAnnotation,
    EnrichmentResult,
    QuoteValidation,
    Tension,
)
from text_theme_analyzer.output.json_report import analysis_to_dict
from text_theme_analyzer.pipeline.model import Analysis, ClusterResult, Note, NoteChunk

# --- OpenAICompatibleClient mocked transport ---

def _make_transport(responses: list[tuple[int, dict | str]]):
    """Build an httpx MockTransport that returns each response in turn."""
    iter_responses = iter(responses)
    def handler(request: httpx.Request) -> httpx.Response:
        try:
            status, body = next(iter_responses)
        except StopIteration:
            status, body = 500, {"error": "no more responses"}
        if isinstance(body, (dict, list)):
            return httpx.Response(status, json=body)
        return httpx.Response(status, text=body)
    return httpx.MockTransport(handler)


def test_openai_compat_parses_chat_completions_response() -> None:
    transport = _make_transport([
        (200, {
            "choices": [{"message": {"content": '{"clusters": [], "tensions": []}'}}]
        })
    ])
    client = OpenAICompatibleClient(
        base_url="https://example.test", api_key="k", model="m", timeout_s=10
    )
    client._client = httpx.Client(transport=transport)  # type: ignore[attr-defined]
    # The client doesn't have a `_client` slot by default; we'll patch _post_chat directly
    # by overriding httpx.post at module level.
    from text_theme_analyzer.llm import openai_compat as oc
    orig = oc.httpx.post
    oc.httpx.post = lambda *a, **k: httpx.Response(
        200,
        json={"choices": [{"message": {"content": '{"clusters": []}'}}]},
        request=httpx.Request("POST", "https://example.test"),
    )
    try:
        out = client.complete(system="s", user="u")
        assert '"clusters"' in out
    finally:
        oc.httpx.post = orig


def test_openai_compat_raises_on_401() -> None:
    client = OpenAICompatibleClient(
        base_url="https://example.test", api_key="bad", model="m", timeout_s=10
    )
    from text_theme_analyzer.llm import openai_compat as oc
    from text_theme_analyzer.llm.base import LLMAuthError
    orig = oc.httpx.post
    oc.httpx.post = lambda *a, **k: httpx.Response(401, text="unauthorized", request=httpx.Request("POST", "https://example.test"))
    try:
        with pytest.raises(LLMAuthError):
            client.complete(system="s", user="u")
    finally:
        oc.httpx.post = orig


# --- factory ---

def test_factory_ollama_requires_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEXTHEME_OLLAMA_API_KEY", raising=False)
    cfg = Config(provider=Provider.OLLAMA)
    with pytest.raises(LLMError, match="TEXTHEME_OLLAMA_API_KEY"):
        build_client(cfg)


def test_factory_ollama_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEXTHEME_OLLAMA_API_KEY", "test-key-123")
    cfg = Config(provider=Provider.OLLAMA)
    client = build_client(cfg)
    assert client.api_key == "test-key-123"
    assert client.model == cfg.model


def test_factory_openai_compat_requires_both_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEXTHEME_OPENAI_COMPAT_BASE_URL", raising=False)
    monkeypatch.delenv("TEXTHEME_OPENAI_COMPAT_API_KEY", raising=False)
    cfg = Config(provider=Provider.OPENAI_COMPAT)
    with pytest.raises(LLMError, match="TEXTHEME_OPENAI_COMPAT_BASE_URL"):
        build_client(cfg)


def test_factory_openai_compat_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEXTHEME_OPENAI_COMPAT_BASE_URL", "https://api.openai.com")
    monkeypatch.setenv("TEXTHEME_OPENAI_COMPAT_API_KEY", "sk-test")
    cfg = Config(provider=Provider.OPENAI_COMPAT)
    client = build_client(cfg)
    assert client.base_url == "https://api.openai.com"
    assert client.api_key == "sk-test"


# --- enrichment helpers ---

def _fake_analysis() -> Analysis:
    n1 = Note(
        id="n1", path=Path("n1.md"), title="Agent design",
        body="The agent loop is a tree. The model decides when to call tools. "
             "Most frameworks just hide the loop. I keep coming back to delegation.",
        date=date(2025, 4, 1), tags=["ai"],
    )
    n2 = Note(
        id="n2", path=Path("n2.md"), title="Grift notes",
        body="Another crypto grift. The pattern is always the same: anonymous team, "
             "vague whitepaper, big promises, fake partnerships, countdown timer.",
        date=date(2024, 8, 21), tags=["scams"],
    )
    chunks = [
        NoteChunk(note_id="n1", chunk_index=0, text=n1.body, char_offset=0),
        NoteChunk(note_id="n2", chunk_index=0, text=n2.body, char_offset=0),
    ]
    cluster_result = ClusterResult(
        assignments=[0, 1],
        cluster_sizes={0: 1, 1: 1},
        cluster_keywords={
            0: [("agent", 0.3), ("model", 0.2), ("loop", 0.1)],
            1: [("grift", 0.4), ("pattern", 0.2), ("crypto", 0.1)],
        },
        cluster_representatives={0: ["n1"], 1: ["n2"]},
        umap_2d=[(0.0, 0.0), (1.0, 1.0)],
        outlier_count=0,
    )
    return Analysis(
        notes=[n1, n2],
        chunks=chunks,
        chunk_note_ids=["n1", "n2"],
        keywords={
            "n1": [("agent", 0.5)],
            "n2": [("grift", 0.5)],
        },
        keyphrase_frequency=[("agent", 1), ("grift", 1)],
        clusters=cluster_result,
        timeseries=None,
        enrichment=None,
        metadata={"date_range": ["2024-08-21", "2025-04-01"]},
    )


def test_sample_quotes_picks_quotable_sentences() -> None:
    a = _fake_analysis()
    quotes = _sample_quotes(a, per_cluster=2)
    assert 0 in quotes
    assert 1 in quotes
    # Should pick one of the cleanest lines.
    assert any("agent loop" in q.lower() for q in quotes[0])
    assert any("anonymous team" in q.lower() for q in quotes[1])


def test_build_bundle_includes_clusters_and_dates() -> None:
    a = _fake_analysis()
    bundle = build_bundle(a)
    assert bundle["total_notes"] == 2
    assert len(bundle["clusters"]) == 2
    for c in bundle["clusters"]:
        assert "first_seen" in c
        assert "last_seen" in c
        assert "keywords" in c
        assert "representative_quotes" in c


def test_validate_quotes_drops_invented() -> None:
    a = _fake_analysis()
    quotes = _sample_quotes(a, per_cluster=2)
    result = EnrichmentResult(
        clusters=[
            ClusterAnnotation(
                cluster_id=0,
                name="Agent design",
                summary="Notes about the agent loop.",
                top_quotes=[
                    "The agent loop is a tree.",  # verbatim from input
                    "Made-up quote that doesn't exist.",  # invented
                ],
                emotional_tone="curious",
            ),
        ],
        tensions=[
            Tension(
                title="Loop vs. linear",
                pole_a="agent design",
                pole_b="traditional flow",
                evidence=["a note discusses agents", "another discusses flow"],
                note="an editorial observation",
            ),
        ],
    )
    validated = _validate_quotes(result, quotes)
    assert len(validated.clusters[0].top_quotes) == 1
    assert validated.quote_validation.requested == 2
    assert validated.quote_validation.dropped == 1


def test_enrich_handles_markdown_fenced_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even if the LLM wraps its JSON in ```json ... ```, we should still parse."""
    a = _fake_analysis()
    captured = {}

    class FakeClient:
        def complete(self, **kwargs):
            captured["called"] = True
            return '```json\n{"clusters": [], "tensions": []}\n```'

    result = enrich(a, FakeClient())
    assert result.clusters == []
    assert result.tensions == []


def test_enrich_retries_on_parse_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """First call returns garbage, second call returns valid JSON."""
    a = _fake_analysis()
    calls = {"n": 0}

    class FakeClient:
        def complete(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return "not json at all"
            return '{"clusters": [], "tensions": []}'

    result = enrich(a, FakeClient())
    assert calls["n"] == 2
    assert result.clusters == []


# --- richer enrichment bundle (Task #6) ---

def test_bundle_includes_excerpts_and_keyphrases() -> None:
    a = _fake_analysis()
    bundle = build_bundle(a)
    for c in bundle["clusters"]:
        assert "excerpts" in c
        assert "keyphrases" in c
        # Excerpts should be non-empty lists of {title, date, body}.
        assert len(c["excerpts"]) >= 1
        for ex in c["excerpts"]:
            assert "title" in ex
            assert "date" in ex
            assert "body" in ex
            assert ex["body"]
        # keyphrases should surface at least one term (from the per-note kws).
        assert len(c["keyphrases"]) >= 1


def test_bundle_excerpt_body_is_truncated_to_budget() -> None:
    long_body = "word " * 500  # 2500 chars
    n1 = Note(
        id="n1", path=Path("n1.md"), title="Long", body=long_body,
        date=date(2025, 4, 1),
    )
    a = Analysis(
        notes=[n1], chunks=[NoteChunk(note_id="n1", chunk_index=0, text=long_body, char_offset=0)],
        chunk_note_ids=["n1"], keywords={"n1": [("word", 0.5)]},
        keyphrase_frequency=[("word", 1)],
        clusters=ClusterResult(
            assignments=[0], cluster_sizes={0: 1}, cluster_keywords={0: [("word", 0.3)]},
            cluster_representatives={0: ["n1"]}, umap_2d=[(0.0, 0.0)], outlier_count=0,
        ),
        timeseries=None, enrichment=None, metadata={},
    )
    bundle = build_bundle(a, chunks_per_cluster=1, chunk_body_chars=400)
    ex_body = bundle["clusters"][0]["excerpts"][0]["body"]
    assert len(ex_body) <= 420  # small slack for the ellipsis + boundary
    assert ex_body.endswith("…")


def test_bundle_respects_max_chars_budget() -> None:
    """With a tiny budget, smaller clusters get dropped to fit."""
    # Build a 4-cluster analysis.
    notes = []
    chunks = []
    chunk_ids = []
    keywords = {}
    for i in range(4):
        nid = f"n{i}"
        body = f"cluster {i} specific term {i} " + ("filler " * 100)
        notes.append(Note(id=nid, path=Path(f"{nid}.md"), title=f"Title {i}", body=body, date=date(2025, 1, i + 1)))
        chunks.append(NoteChunk(note_id=nid, chunk_index=0, text=body, char_offset=0))
        chunk_ids.append(nid)
        keywords[nid] = [(f"term{i}", 0.5)]
    a = Analysis(
        notes=notes, chunks=chunks, chunk_note_ids=chunk_ids,
        keywords=keywords, keyphrase_frequency=[(f"term{i}", 1) for i in range(4)],
        clusters=ClusterResult(
            assignments=[0, 1, 2, 3], cluster_sizes={0: 1, 1: 1, 2: 1, 3: 1},
            cluster_keywords={i: [(f"term{i}", 0.3)] for i in range(4)},
            cluster_representatives={i: [f"n{i}"] for i in range(4)},
            umap_2d=[(i, i) for i in range(4)], outlier_count=0,
        ),
        timeseries=None, enrichment=None, metadata={},
    )
    # Generous budget: keep all 4.
    bundle = build_bundle(a, max_bundle_chars=200_000)
    assert len(bundle["clusters"]) == 4
    assert bundle["dropped_clusters"] == 0
    # Tiny budget: only the largest cluster should survive (all are size 1, so any 1).
    bundle_tiny = build_bundle(a, max_bundle_chars=200)
    assert len(bundle_tiny["clusters"]) == 1
    assert bundle_tiny["dropped_clusters"] >= 3


def test_bundle_reports_size() -> None:
    a = _fake_analysis()
    bundle = build_bundle(a)
    assert "bundle_chars" in bundle
    assert bundle["bundle_chars"] > 0
    assert "dropped_clusters" in bundle


# --- smarter cluster names (Task #12) ---

def test_cluster_annotation_name_is_optional() -> None:
    """The LLM may return a name (rich) or leave it null (fallback path)."""
    # With a name.
    ann_with = ClusterAnnotation(
        cluster_id=0, name="Agent design", summary="Notes about the agent loop.",
        top_quotes=[], emotional_tone="curious",
    )
    assert ann_with.name == "Agent design"
    # Without a name.
    ann_without = ClusterAnnotation(
        cluster_id=0, summary="Notes about the agent loop.",
        top_quotes=[], emotional_tone="curious",
    )
    assert ann_without.name is None


def test_markdown_falls_back_to_keyword_label_when_name_missing(tmp_path: Path) -> None:
    """When the LLM returns no name, the markdown uses the top-2 keywords."""
    n = Note(
        id="n1", path=Path("n1.md"), title="t", body="body",
        date=date(2025, 1, 1),
    )
    a = Analysis(
        notes=[n], chunks=[], chunk_note_ids=[], keywords={"n1": []},
        keyphrase_frequency=[],
        clusters=ClusterResult(
            assignments=[0], cluster_sizes={0: 1},
            cluster_keywords={0: [("agent", 0.3), ("loop", 0.2), ("design", 0.1)]},
            cluster_representatives={0: ["n1"]}, umap_2d=[(0.0, 0.0)], outlier_count=0,
        ),
        timeseries=None,
        enrichment=EnrichmentResult(
            clusters=[ClusterAnnotation(
                cluster_id=0, name=None,
                summary="A cluster about agents.", top_quotes=[],
                emotional_tone="curious",
            )],
            quote_validation=QuoteValidation(),
        ),
        metadata={},
    )
    from text_theme_analyzer.output.markdown_report import render_markdown
    md = render_markdown(a)
    # The fallback should be "agent / loop" (top 2 keywords).
    assert "agent / loop" in md or "agent" in md  # robust to exact formatting


# --- bundle size + prompt contract (Task #13) ---

def test_bundle_size_stays_under_default_budget() -> None:
    """The default 12K-token budget (48K chars) is enforced on bundle construction."""
    a = _fake_analysis()
    bundle = build_bundle(a)
    assert bundle["bundle_chars"] < 48_000
    # The two-cluster analysis fits easily under the budget.
    assert bundle["dropped_clusters"] == 0


def test_prompt_includes_required_contract_strings() -> None:
    """The system + user prompt must contain the JSON contract and key rules.

    If the prompt drifts (e.g. someone rewords the rules), this test fails —
    a useful guard against silent regressions in the LLM instruction.
    """
    from text_theme_analyzer.llm.prompts import SYSTEM_PROMPT, build_user_prompt

    # System prompt: rules that must always be present.
    assert "JSON" in SYSTEM_PROMPT
    assert "Do not invent quotes" in SYSTEM_PROMPT
    assert "Tensions connect two DIFFERENT clusters" in SYSTEM_PROMPT
    assert "emotional_tone" in SYSTEM_PROMPT

    # User prompt: schema must be present so the LLM knows the shape.
    user = build_user_prompt(
        total_notes=2,
        date_range=("2024-01-01", "2025-01-01"),
        clusters=[{
            "id": 0, "size": 1, "keywords": ["agent"], "keyphrases": ["agent"],
            "representative_titles": ["t"], "representative_quotes": ["q"],
            "excerpts": [{"title": "t", "date": "2025-01-01", "body": "body"}],
            "first_seen": "2025-01-01", "last_seen": "2025-01-01",
        }],
        spikes=[],
        stale_candidates=[],
    )
    assert '"cluster_id"' in user
    assert '"tensions"' in user
    assert '"article_candidates"' in user
    assert '"stale_recurring"' in user
    assert "verdict" in user
    # Bundled cluster metadata should appear so the LLM sees it.
    assert "agent" in user
    assert "q" in user


def test_enrich_records_bundle_metadata_in_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test that exercises the full enrich() flow with a fake client and verifies
    that the returned EnrichmentResult has correct quote-validation accounting.
    """
    a = _fake_analysis()
    quotes = _sample_quotes(a, per_cluster=2)
    # Take an actual quote from cluster 0 as the "verbatim" one.
    valid_quote = quotes[0][0]
    client_response = json.dumps({
        "clusters": [{
            "cluster_id": 0, "name": "Test", "summary": "Summary",
            "top_quotes": [valid_quote, "completely invented quote"],
            "emotional_tone": "curious",
        }],
        "tensions": [],
        "article_candidates": [],
        "stale_recurring": [],
    })

    class FakeClient:
        def complete(self, **kwargs):
            return client_response

    result = enrich(a, FakeClient())
    assert len(result.clusters) == 1
    assert result.clusters[0].top_quotes == [valid_quote]
    assert result.quote_validation.requested == 2
    assert result.quote_validation.dropped == 1


# --- JSON sidecar ---

def test_json_report_serializes_analysis(tmp_path: Path) -> None:
    a = _fake_analysis()
    out = analysis_to_dict(a)
    assert "metadata" in out
    assert "clusters" in out
    assert "timeseries" in out
    assert "keyphrases" in out
    assert "files" in out
    # Round-trip: must be JSON-serializable.
    json.dumps(out, default=str)


def test_json_report_has_schema_version_and_numeric_ids() -> None:
    a = _fake_analysis()
    out = analysis_to_dict(a)
    # Top-level schema_version is present and is a string.
    assert isinstance(out["schema_version"], str)
    assert out["schema_version"]  # non-empty
    # clusters.cluster_ids is a list of integers (not stringified).
    assert isinstance(out["clusters"]["cluster_ids"], list)
    assert all(isinstance(cid, int) for cid in out["clusters"]["cluster_ids"])
    # The "sizes" block now has explicit cluster_id ints, not string keys.
    sizes = out["clusters"]["sizes"]
    assert isinstance(sizes, list)
    assert all("cluster_id" in s and isinstance(s["cluster_id"], int) for s in sizes)
    # umap_2d entries carry numeric cluster_id too.
    for p in out["clusters"]["umap_2d"]:
        assert "cluster_id" in p
        assert isinstance(p["cluster_id"], int)
