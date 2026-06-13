"""Build the LLM bundle from an Analysis and call the enrichment endpoint.

Steps:
1. Sample 2-3 strong quotes per cluster (prefer long, line-like sentences).
2. Pull the top-N chunk bodies per cluster (truncated to a per-chunk budget).
3. Build the bundle, enforce a global character cap so the prompt fits the
   model's context window.
4. Send the bundle to the LLM via the configured client.
5. Parse + validate against Pydantic schema; retry once on parse error
   with the error message appended + lower temperature.
6. Post-validate returned quotes against the input quote pool (drop invented ones).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date

from pydantic import ValidationError

from text_theme_analyzer.llm.base import LLMClient, LLMParseError
from text_theme_analyzer.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from text_theme_analyzer.llm.schemas import (
    ClusterAnnotation,
    EnrichmentResult,
    QuoteValidation,
)
from text_theme_analyzer.pipeline.model import Analysis

# Rough per-character-to-token estimate. ~4 chars per token for English.
# Used to size the bundle so it fits comfortably in a 16K context window.
CHARS_PER_TOKEN = 4

# Per-chunk body budget (chars). Long enough to convey the chunk's content,
# short enough to keep the bundle compact.
DEFAULT_CHUNK_BODY_CHARS = 1200

# Per-cluster max chunks included in the LLM bundle.
DEFAULT_CHUNKS_PER_CLUSTER = 3

# Global cap on bundle character size (rough proxy for ~12K input tokens).
DEFAULT_MAX_BUNDLE_CHARS = 12_000 * CHARS_PER_TOKEN


def _split_into_quote_candidates(text: str, min_words: int = 6, max_words: int = 40) -> list[str]:
    """Heuristic: split body into sentences, keep 'quotable' ones."""
    # Naive sentence splitter: split on `.`, `!`, `?`, or newlines, then strip.
    raw = re.split(r"(?<=[.!?])\s+|\n+", text)
    out: list[str] = []
    for r in raw:
        r = r.strip().strip("\"'`")
        words = r.split()
        if min_words <= len(words) <= max_words:
            out.append(r)
    return out


def _sample_quotes(analysis: Analysis, per_cluster: int = 3) -> dict[int, list[str]]:
    """Pick up to `per_cluster` strong quotes per cluster from the chunks in that cluster."""
    if analysis.clusters is None:
        return {}
    note_by_id = {n.id: n for n in analysis.notes}
    # Build a map chunk_index -> cluster_id.
    chunk_to_cluster = dict(enumerate(analysis.clusters.assignments))
    # Group chunks by cluster.
    by_cluster: dict[int, list[int]] = {}
    for chunk_idx, cid in chunk_to_cluster.items():
        by_cluster.setdefault(cid, []).append(chunk_idx)
    out: dict[int, list[str]] = {}
    for cid, indices in by_cluster.items():
        if cid == -1:
            continue
        seen_text: set[str] = set()
        picked: list[str] = []
        for idx in indices:
            note_id = analysis.chunk_note_ids[idx]
            note = note_by_id.get(note_id)
            if note is None:
                continue
            for cand in _split_into_quote_candidates(analysis.chunks[idx].text):
                norm = re.sub(r"\s+", " ", cand.lower()).strip()
                if norm in seen_text:
                    continue
                seen_text.add(norm)
                picked.append(cand)
                if len(picked) >= per_cluster:
                    break
            if len(picked) >= per_cluster:
                break
        out[cid] = picked
    return out


def _rep_titles(analysis: Analysis) -> dict[int, list[str]]:
    if analysis.clusters is None:
        return {}
    note_by_id = {n.id: n for n in analysis.notes}
    out: dict[int, list[str]] = {}
    for cid, ids in analysis.clusters.cluster_representatives.items():
        out[cid] = [note_by_id[i].title for i in ids if i in note_by_id]
    return out


def _top_chunks_per_cluster(
    analysis: Analysis,
    *,
    per_cluster: int = DEFAULT_CHUNKS_PER_CLUSTER,
    body_chars: int = DEFAULT_CHUNK_BODY_CHARS,
) -> dict[int, list[dict]]:
    """Pick the top-N chunks per cluster (by cluster size order) and trim each to `body_chars`.

    Returns {cluster_id: [{"title": ..., "body": ..., "date": ...}, ...]}.
    Chunks are taken in the order they appear in the cluster (reps first if
    available, then the rest of the cluster).
    """
    if analysis.clusters is None:
        return {}
    note_by_id = {n.id: n for n in analysis.notes}
    # Map chunk index -> cluster id, skip outliers.
    chunk_to_cluster: dict[int, int] = {}
    for idx, cid in enumerate(analysis.clusters.assignments):
        if cid != -1:
            chunk_to_cluster[idx] = cid
    by_cluster: dict[int, list[int]] = {}
    for idx, cid in chunk_to_cluster.items():
        by_cluster.setdefault(cid, []).append(idx)

    # Rank chunks within a cluster: prefer representative chunks first, then
    # in the order they were assigned.
    rep_chunks: dict[int, set[int]] = {}
    for cid, rep_note_ids in analysis.clusters.cluster_representatives.items():
        rep_chunks[cid] = {
            i for i, nid in enumerate(analysis.chunk_note_ids)
            if nid in set(rep_note_ids) and chunk_to_cluster.get(i) == cid
        }

    out: dict[int, list[dict]] = {}
    for cid, indices in by_cluster.items():
        ordered = sorted(indices, key=lambda i: (i not in rep_chunks.get(cid, set()), i))
        picked: list[dict] = []
        for idx in ordered[:per_cluster]:
            note_id = analysis.chunk_note_ids[idx]
            note = note_by_id.get(note_id)
            if note is None:
                continue
            body = analysis.chunks[idx].text.strip()
            if len(body) > body_chars:
                # Truncate at the nearest sentence/word boundary so we don't
                # cut mid-word.
                cut = body[:body_chars]
                # Try to back off to the last sentence terminator.
                for term in (". ", "! ", "? ", "\n"):
                    pos = cut.rfind(term)
                    if pos > body_chars * 0.5:
                        cut = cut[: pos + 1]
                        break
                body = cut.rstrip() + "…"
            picked.append({
                "title": note.title,
                "date": note.date.isoformat() if note.date else "?",
                "body": body,
            })
        out[cid] = picked
    return out


def _cluster_keyphrases(analysis: Analysis, top_n: int = 8) -> dict[int, list[str]]:
    """Per-cluster keyphrase list from analysis.keywords (aggregated across notes in the cluster)."""
    if analysis.clusters is None:
        return {}
    note_to_cluster: dict[str, int] = {}
    for idx, cid in enumerate(analysis.clusters.assignments):
        if cid == -1:
            continue
        nid = analysis.chunk_note_ids[idx]
        # First cluster wins for a note (a note may span multiple clusters).
        note_to_cluster.setdefault(nid, cid)
    out: dict[int, list[str]] = {}
    for nid, cid in note_to_cluster.items():
        phrases = [p for p, _ in analysis.keywords.get(nid, [])]
        out.setdefault(cid, [])
        for p in phrases:
            if p not in out[cid]:
                out[cid].append(p)
                if len(out[cid]) >= top_n:
                    break
    return out


def _cluster_tags(analysis: Analysis, top_n: int = 20) -> dict[int, list[str]]:
    """Per-cluster tag distribution, derived from notes' `tags` field.

    For each cluster, count tag occurrences across all member notes
    (counting a tag once per note even if the note appears in multiple
    chunks — the cluster-level signal is "which themes the writer
    associated with this cluster", not raw tag-chunk counts). Returns
    the top-N tags sorted by frequency desc.

    Returns {} when the corpus has no tagged notes, and {} per cluster
    when a cluster has no tagged member notes.

    Note: the same tag information exists as an `(M, N)` matrix built
    by `pipeline.clustering.build_tag_matrix` and consumed by
    `cluster_chunks` (when `tag_weight > 0`). We read strings directly
    from `Note.tags` here because the LLM wants human-readable tag
    *names*, not column indices. The matrix is the right primitive for
    clustering weights; this is the right primitive for the prompt.
    """
    if analysis.clusters is None:
        return {}
    notes_by_id = {n.id: n for n in analysis.notes}
    # Map each note to its primary cluster (first non-outlier cluster wins,
    # matching `_cluster_keyphrases` and the rest of the bundle code).
    note_to_cluster: dict[str, int] = {}
    for chunk_idx, cid in enumerate(analysis.clusters.assignments):
        if cid == -1:
            continue
        nid = analysis.chunk_note_ids[chunk_idx]
        note_to_cluster.setdefault(nid, cid)

    cluster_counts: dict[int, Counter] = {}
    for nid, cid in note_to_cluster.items():
        note = notes_by_id.get(nid)
        if note is None or not note.tags:
            continue
        # Count each tag at most once per note (avoids double-counting
        # when a note spans multiple chunks in the same cluster).
        for t in set(note.tags):
            cluster_counts.setdefault(cid, Counter())[t] += 1

    return {
        cid: [t for t, _ in counts.most_common(top_n)]
        for cid, counts in cluster_counts.items()
    }


def _truncate_bundle_for_budget(
    cluster_blocks: list[dict],
    *,
    max_chars: int,
    chunk_body_chars: int,
) -> tuple[list[dict], int]:
    """Two-pass shrink: first drop whole clusters from the tail, then shrink
    chunk bodies, then drop individual chunks. Returns (trimmed_blocks, dropped_clusters)."""
    total = sum(len(json.dumps(b, default=str)) for b in cluster_blocks)
    dropped = 0
    if total <= max_chars:
        return cluster_blocks, dropped

    # Pass 1: drop smallest clusters (last in the sorted list).
    while len(cluster_blocks) > 1:
        total = sum(len(json.dumps(b, default=str)) for b in cluster_blocks)
        if total <= max_chars:
            break
        cluster_blocks = cluster_blocks[:-1]
        dropped += 1
    if total <= max_chars:
        return cluster_blocks, dropped

    # Pass 2: shrink chunk bodies.
    shrunk = chunk_body_chars
    while shrunk > 200:
        shrunk = int(shrunk * 0.6)
        for b in cluster_blocks:
            for ex in b.get("excerpts", []):
                ex["body"] = ex["body"][:shrunk]
        total = sum(len(json.dumps(b, default=str)) for b in cluster_blocks)
        if total <= max_chars:
            return cluster_blocks, dropped

    # Pass 3: drop the last excerpt from each cluster.
    for b in cluster_blocks:
        if b.get("excerpts"):
            b["excerpts"] = b["excerpts"][:-1]
    return cluster_blocks, dropped


def _first_last_seen(analysis: Analysis) -> dict[int, tuple[date | None, date | None]]:
    """For each cluster, find the first and last note date among its notes."""
    if analysis.clusters is None:
        return {}
    n2c = {}
    for chunk_idx, cid in enumerate(analysis.clusters.assignments):
        if cid == -1:
            continue
        note_id = analysis.chunk_note_ids[chunk_idx]
        if note_id not in n2c:
            n2c[note_id] = cid
    note_by_id = {n.id: n for n in analysis.notes}
    by_cluster: dict[int, list[date]] = {}
    for note_id, cid in n2c.items():
        d = note_by_id[note_id].date
        if d is None:
            continue
        by_cluster.setdefault(cid, []).append(d)
    out: dict[int, tuple[date | None, date | None]] = {}
    for cid, dates in by_cluster.items():
        out[cid] = (min(dates), max(dates))
    return out


def build_bundle(
    analysis: Analysis,
    *,
    top_n_clusters: int = 20,
    chunks_per_cluster: int = DEFAULT_CHUNKS_PER_CLUSTER,
    chunk_body_chars: int = DEFAULT_CHUNK_BODY_CHARS,
    max_bundle_chars: int = DEFAULT_MAX_BUNDLE_CHARS,
) -> dict:
    """Construct the structured bundle sent to the LLM.

    Includes cluster keywords, sizes, representative titles, sampled quotes,
    top keyphrases per cluster, and a few representative chunk bodies
    (truncated). If the bundle exceeds `max_bundle_chars`, smaller clusters
    are dropped and chunk bodies shrunk to fit.
    """
    if analysis.clusters is None:
        raise ValueError("Clustering must run before enrichment.")
    quotes = _sample_quotes(analysis)
    titles = _rep_titles(analysis)
    seen = _first_last_seen(analysis)
    excerpts = _top_chunks_per_cluster(
        analysis,
        per_cluster=chunks_per_cluster,
        body_chars=chunk_body_chars,
    )
    keyphrases = _cluster_keyphrases(analysis)
    tags = _cluster_tags(analysis)

    # Sort clusters by size desc, take top N.
    cluster_ids = sorted(
        analysis.clusters.cluster_keywords.keys(),
        key=lambda c: analysis.clusters.cluster_sizes.get(c, 0),
        reverse=True,
    )[:top_n_clusters]

    clusters_for_prompt = []
    for cid in cluster_ids:
        kws = [w for w, _ in analysis.clusters.cluster_keywords.get(cid, [])[:8]]
        fl = seen.get(cid, (None, None))
        # Stable key for the "promote to project" affordance. Format:
        # "<cluster_id>:<last_seen_iso>". If the cluster comes back to
        # life, last_seen shifts and the key changes — the prior entry
        # in the target file becomes a historical artifact (informative).
        last_seen_str = fl[1].isoformat() if fl[1] else "unknown"
        promote_key = f"{cid}:{last_seen_str}"
        clusters_for_prompt.append({
            "id": cid,
            "size": analysis.clusters.cluster_sizes.get(cid, 0),
            "keywords": kws,
            "keyphrases": keyphrases.get(cid, []),
            "tags": tags.get(cid, []),
            "representative_titles": titles.get(cid, []),
            "representative_quotes": quotes.get(cid, []),
            "excerpts": excerpts.get(cid, []),
            "first_seen": fl[0].isoformat() if fl[0] else "?",
            "last_seen": last_seen_str,
            "promote_key": promote_key,
        })

    # Enforce bundle size budget.
    clusters_for_prompt, dropped_clusters = _truncate_bundle_for_budget(
        clusters_for_prompt,
        max_chars=max_bundle_chars,
        chunk_body_chars=chunk_body_chars,
    )

    spikes_for_prompt = [
        {
            "cluster_id": s.cluster_id,
            "bucket": s.bucket.isoformat(),
            "count": s.count,
            "delta": s.delta,
        }
        for s in (analysis.timeseries.spikes if analysis.timeseries else [])
    ]
    stale_for_prompt = [
        {
            "cluster_id": s.cluster_id,
            "first_seen": s.first_seen.isoformat(),
            "last_seen": s.last_seen.isoformat(),
            "frequency": s.frequency,
        }
        for s in (analysis.timeseries.stale if analysis.timeseries else [])
    ]

    date_range: tuple[str | None, str | None] = (None, None)
    if "date_range" in analysis.metadata:
        date_range = (analysis.metadata["date_range"][0], analysis.metadata["date_range"][1])

    # T1.2a: pass configured project-board sections so the LLM can pick a
    # `target_section` for promote_to_project verdicts. Empty list means
    # no hint is added to the prompt.
    promote_sections: list[str] = []
    if "config" in analysis.metadata:
        promote_sections = list(analysis.metadata["config"].get("promote_sections") or [])

    bundle_size = sum(len(json.dumps(b, default=str)) for b in clusters_for_prompt)
    return {
        "total_notes": len(analysis.notes),
        "date_range": date_range,
        "clusters": clusters_for_prompt,
        "spikes": spikes_for_prompt,
        "stale_candidates": stale_for_prompt,
        "bundle_chars": bundle_size,
        "dropped_clusters": dropped_clusters,
        "promote_sections": promote_sections,
        "_quotes_by_cluster": quotes,  # used locally for validation, not sent
    }


def _parse_with_retry(
    client: LLMClient,
    system: str,
    user: str,
    *,
    max_retries: int = 2,
) -> EnrichmentResult:
    """Call the LLM; on parse error, retry once with the error message appended."""
    last_error: Exception | None = None
    for attempt in range(max_retries):
        temperature = 0.2 if attempt == 0 else 0.0
        try:
            text = client.complete(
                system=system,
                user=user,
                temperature=temperature,
                json_mode=True,
                # Output budget for the JSON response. The model produces
                # ~12-25KB of JSON for a real-sized corpus (11+ clusters,
                # 8+ tensions, 8+ article candidates with verbose prose).
                # 12288 covers that; if you see "Unterminated string" or
                # "Expecting value" parse errors, raise this.
                max_tokens=12288,
            )
        except Exception as e:
            last_error = e
            continue
        # The model may return a JSON object possibly wrapped in ```json ... ```.
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            data = json.loads(cleaned)
            return EnrichmentResult.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            user = user + f"\n\nYour previous response did not validate:\n{e}\nPlease return a corrected JSON object that matches the schema."
    raise LLMParseError(
        f"LLM response failed to parse after {max_retries} attempts: {last_error}"
    )


def _validate_quotes(
    result: EnrichmentResult,
    quotes_by_cluster: dict[int, list[str]],
) -> EnrichmentResult:
    """Drop any LLM-returned quote that doesn't appear in the input quotes for that cluster."""
    requested = 0
    dropped = 0
    kept_clusters: list[ClusterAnnotation] = []
    for ann in result.clusters:
        cluster_quotes = quotes_by_cluster.get(ann.cluster_id, [])
        norm_pool = {re.sub(r"\s+", " ", q.lower()).strip() for q in cluster_quotes}
        kept: list[str] = []
        for q in ann.top_quotes:
            requested += 1
            qn = re.sub(r"\s+", " ", q.lower()).strip()
            if qn in norm_pool:
                kept.append(q)
            else:
                dropped += 1
        kept_clusters.append(ann.model_copy(update={"top_quotes": kept}))
    return result.model_copy(
        update={
            "clusters": kept_clusters,
            "quote_validation": QuoteValidation(requested=requested, dropped=dropped),
        }
    )


def enrich(analysis: Analysis, client: LLMClient) -> EnrichmentResult:
    """Build the bundle, call the LLM, validate, and return the EnrichmentResult.

    Note: bundle size + dropped cluster info is left in the bundle dict
    but consumed locally (the EnrichmentResult Pydantic model does not carry
    it). Callers that want the bundle metadata can call `build_bundle()`
    directly.
    """
    bundle = build_bundle(analysis)
    quotes_by_cluster = bundle.pop("_quotes_by_cluster")
    user_prompt = build_user_prompt(
        total_notes=bundle["total_notes"],
        date_range=bundle["date_range"],
        clusters=bundle["clusters"],
        spikes=bundle["spikes"],
        stale_candidates=bundle["stale_candidates"],
        promote_sections=bundle.get("promote_sections"),
    )
    result = _parse_with_retry(client, SYSTEM_PROMPT, user_prompt)
    return _validate_quotes(result, quotes_by_cluster)
