"""M5 orchestrator: ingest → preprocess → keywords → embeddings → clusters → timeseries → tone → LLM enrichment."""

from __future__ import annotations

from typing import Any

from text_theme_analyzer.config import Config
from text_theme_analyzer.pipeline.clustering import (
    build_tag_matrix,
    cluster_chunks,
    note_to_cluster,
)
from text_theme_analyzer.pipeline.embeddings import EmbeddingCache, embed_corpus
from text_theme_analyzer.pipeline.ingest import load_notes
from text_theme_analyzer.pipeline.keywords import (
    _aggregate_frequency,
    extract_keyphrases,
    keyphrases_for_notes,
)
from text_theme_analyzer.pipeline.model import Analysis, ClusterResult
from text_theme_analyzer.pipeline.preprocess import preprocess_note
from text_theme_analyzer.pipeline.timeseries import build_timeseries
from text_theme_analyzer.pipeline.tone import tone_over_time
from text_theme_analyzer.utils.dates import has_authoritative_date
from text_theme_analyzer.utils.progress import log


def _filter_by_date(notes: list, since, until) -> list:
    if since is None and until is None:
        return notes
    out = []
    for n in notes:
        if n.date is None:
            continue
        if since and n.date < since:
            continue
        if until and n.date > until:
            continue
        out.append(n)
    return out


def run(config: Config) -> Analysis:
    """Run the M5 pipeline: ingest, keywords, embeddings, clusters, time-series, tone, LLM enrichment."""
    log(f"[ingest] scanning {config.input_path}", quiet=config.quiet)
    notes = load_notes(
        config.input_path,
        include=config.include,
        exclude=config.exclude,
    )
    notes = _filter_by_date(notes, config.since, config.until)
    log(f"[ingest] loaded {len(notes)} notes", quiet=config.quiet)

    if config.require_dates:
        undated = [n for n in notes if not has_authoritative_date(n.frontmatter, n.path)]
        if undated:
            paths = "\n  ".join(str(n.path) for n in undated)
            raise ValueError(
                f"--require-dates is set but {len(undated)} note(s) have no explicit date:\n  {paths}\n"
                "Add a `date: YYYY-MM-DD` frontmatter line (or a YYYY-MM-DD filename prefix)."
            )

    all_chunks = []
    chunks_by_note: dict[str, list] = {}
    chunk_note_ids: list[str] = []
    for note in notes:
        chunks = preprocess_note(note)
        chunks_by_note[note.id] = chunks
        all_chunks.extend(chunks)
        for _c in chunks:
            chunk_note_ids.append(note.id)
    log(f"[preprocess] {len(all_chunks)} chunks", quiet=config.quiet)

    per_chunk_kws = extract_keyphrases(
        all_chunks,
        method="keybert",
        top_n=10,
        embedding_model=None,
    )
    keywords = keyphrases_for_notes(notes, chunks_by_note, per_chunk_kws)
    keyphrase_freq = _aggregate_frequency(
        keywords, merge_contained_phrases=config.merge_contained_phrases,
    )
    log(f"[keywords] {len(keyphrase_freq)} unique phrases", quiet=config.quiet)

    cache = None if config.no_cache else EmbeddingCache(
        root=config.cache_dir, model_name=config.embedding_model
    )
    embeddings = embed_corpus(
        [c.text for c in all_chunks],
        model_name=config.embedding_model,
        cache=cache,
        batch_size=32,
        show_progress=not config.quiet,
    )
    log(f"[embed] shape {embeddings.shape}", quiet=config.quiet)

    # Tag matrix (T1.1). Built once, used by both clustering (when
    # tag_weight > 0) and the LLM bundle (always — it needs the per-cluster
    # tag distribution regardless of the clustering weight). When the
    # corpus has no tags, the matrix is (M, 0) and contributes nothing.
    # `tag_columns` is the corpus's top-N tag ordering in frequency-desc
    # order — the hook for per-tag weight tables (T1.1b). It's passed to
    # `cluster_chunks` so individual tags can be scaled independently.
    tag_matrix, tag_columns = build_tag_matrix(notes, all_chunks, top_n_tags=config.top_n_tags)
    if tag_matrix.shape[1] > 0:
        n_tag_weights = len(config.tag_weights)
        log(
            f"[tag] using top {tag_matrix.shape[1]} tags "
            f"(weight={config.tag_weight}, {n_tag_weights} per-tag weights)",
            quiet=config.quiet,
        )

    try:
        clusters = cluster_chunks(
            all_chunks,
            embeddings,
            min_cluster_size=config.min_cluster_size,
            umap_n_neighbors=config.umap_n_neighbors if config.umap_n_neighbors is not None else 15,
            tag_weight=config.tag_weight,
            top_n_tags=config.top_n_tags,
            tag_matrix=tag_matrix,
            tag_columns=tag_columns,
            tag_weights=config.tag_weights,
        )
    except Exception as e:
        log(f"[cluster] skipped (not enough data for UMAP/HDBSCAN): {e}", quiet=config.quiet)
        clusters = ClusterResult(
            assignments=[-1] * len(all_chunks),
            cluster_sizes={},
            cluster_keywords={},
            cluster_representatives={},
            umap_2d=[(float(i), 0.0) for i in range(len(all_chunks))],
            outlier_count=len(all_chunks),
        )
    if clusters is None or (clusters.assignments and len(clusters.assignments) != len(all_chunks)):
        log("[cluster] too few notes to cluster; skipping", quiet=config.quiet)
        clusters = ClusterResult(
            assignments=[-1] * len(all_chunks),
            cluster_sizes={},
            cluster_keywords={},
            cluster_representatives={},
            umap_2d=[(0.0, 0.0)] * len(all_chunks),
            outlier_count=len(all_chunks),
        )
    log(
        f"[cluster] {len(clusters.cluster_keywords)} clusters, "
        f"{clusters.outlier_count} outliers",
        quiet=config.quiet,
    )

    n2c = note_to_cluster(chunk_note_ids, clusters.assignments)
    note_dates = {n.id: n.date for n in notes}
    if n2c:
        timeseries = build_timeseries(
            n2c,
            note_dates,
            bucket="week",
            spike_window=config.spike_window_weeks,
            stale_window=config.stale_window_weeks,
        )
    else:
        timeseries = None
    if timeseries is not None:
        log(
            f"[series] {len(timeseries.spikes)} spikes, {len(timeseries.stale)} stale",
            quiet=config.quiet,
        )

    # Dry-run short-circuit: estimate the LLM bundle size and bail before
    # paying for the LLM call (or for tone scoring, which is cheap but
    # shouldn't run on a preview either).
    if config.dry_run:
        bundle_estimate = _estimate_bundle(notes, all_chunks, chunk_note_ids, clusters, timeseries)
        log(
            f"[dry-run] would call LLM with ~{bundle_estimate['est_tokens']:,} input tokens "
            f"({bundle_estimate['cluster_count']} clusters, "
            f"{bundle_estimate['chunk_count']} chunks, "
            f"{bundle_estimate['excerpt_chars']} excerpt chars)",
            quiet=config.quiet,
        )
        metadata: dict[str, Any] = {
            "input_path": str(config.input_path),
            "note_count": len(notes),
            "chunk_count": len(all_chunks),
            "config": {
                "embedding_model": config.embedding_model,
                "top_n_themes": config.top_n_themes,
                "spike_window_weeks": config.spike_window_weeks,
                "stale_window_weeks": config.stale_window_weeks,
                "min_cluster_size": config.min_cluster_size,
                "umap_n_neighbors": config.umap_n_neighbors,
                "tag_weight": config.tag_weight,
                "top_n_tags": config.top_n_tags,
                "promote_sections": [str(s) for s in config.promote.sections],
            },
            "dry_run": bundle_estimate,
        }
        if notes and notes[0].date is not None:
            dated = [n.date for n in notes if n.date is not None]
            if dated:
                metadata["date_range"] = [min(dated).isoformat(), max(dated).isoformat()]
        return Analysis(
            notes=notes, chunks=all_chunks, chunk_note_ids=chunk_note_ids,
            keywords=keywords, keyphrase_frequency=keyphrase_freq,
            clusters=clusters, timeseries=timeseries, enrichment=None,
            metadata=metadata,
        )

    # Emotional tone over time (lightweight lexicon; M5 stretch).
    tone_data = tone_over_time(
        {n.id: n.body for n in notes},
        note_dates,
        bucket="month",
    )
    log(f"[tone] {len(tone_data)} monthly buckets scored", quiet=config.quiet)

    # LLM enrichment (optional, gracefully degrades).
    enrichment = None
    if not config.no_llm:
        try:
            from text_theme_analyzer.llm.enrichment import enrich
            from text_theme_analyzer.llm.factory import build_client
            client = build_client(config)
            log(f"[llm] calling {config.provider.value} model {config.model}", quiet=config.quiet)
            mid = _make_analysis(notes, all_chunks, chunk_note_ids, keywords, keyphrase_freq, clusters, timeseries, None)
            enrichment = enrich(mid, client)
            log(
                f"[llm] {len(enrichment.clusters)} cluster annotations, "
                f"{len(enrichment.tensions)} tensions, "
                f"{len(enrichment.article_candidates)} article candidates, "
                f"{len(enrichment.stale_recurring)} stale verdicts "
                f"({enrichment.quote_validation.dropped}/{enrichment.quote_validation.requested} quotes dropped)",
                quiet=config.quiet,
            )
        except Exception as e:
            log(f"[llm] skipped (error): {e}", quiet=config.quiet)
            enrichment = None

    metadata: dict[str, Any] = {
        "input_path": str(config.input_path),
        "note_count": len(notes),
        "chunk_count": len(all_chunks),
        "tone_over_time": tone_data,
        "config": {
            "embedding_model": config.embedding_model,
            "top_n_themes": config.top_n_themes,
            "spike_window_weeks": config.spike_window_weeks,
            "stale_window_weeks": config.stale_window_weeks,
            "min_cluster_size": config.min_cluster_size,
            "umap_n_neighbors": config.umap_n_neighbors,
            "tag_weight": config.tag_weight,
            "top_n_tags": config.top_n_tags,
            "promote_sections": [str(s) for s in config.promote.sections],
        },
    }
    if notes and notes[0].date is not None:
        dated = [n.date for n in notes if n.date is not None]
        if dated:
            metadata["date_range"] = [min(dated).isoformat(), max(dated).isoformat()]

    return Analysis(
        notes=notes,
        chunks=all_chunks,
        chunk_note_ids=chunk_note_ids,
        keywords=keywords,
        keyphrase_frequency=keyphrase_freq,
        clusters=clusters,
        timeseries=timeseries,
        enrichment=enrichment,
        metadata=metadata,
    )


def _make_analysis(notes, chunks, chunk_note_ids, keywords, keyphrase_freq, clusters, timeseries, enrichment) -> Analysis:
    """Helper used internally to build an Analysis mid-pipeline for LLM enrichment."""
    return Analysis(
        notes=notes,
        chunks=chunks,
        chunk_note_ids=chunk_note_ids,
        keywords=keywords,
        keyphrase_frequency=keyphrase_freq,
        clusters=clusters,
        timeseries=timeseries,
        enrichment=enrichment,
        metadata={},
    )


def _estimate_bundle(notes, chunks, chunk_note_ids, clusters, timeseries) -> dict:
    """Estimate the size of the LLM bundle that *would* be sent on a real run.

    Cheap: builds a sample bundle with the same shape as build_bundle() and
    measures its serialized size. Avoids importing the enrichment module
    (which pulls in pydantic only when needed for the real call).
    """
    from text_theme_analyzer.llm.enrichment import (
        DEFAULT_CHUNK_BODY_CHARS,
        DEFAULT_CHUNKS_PER_CLUSTER,
        _cluster_keyphrases,
        _first_last_seen,
        _rep_titles,
        _sample_quotes,
        _top_chunks_per_cluster,
    )
    mid = _make_analysis(notes, chunks, chunk_note_ids, {}, [], clusters, timeseries, None)
    quotes = _sample_quotes(mid)
    titles = _rep_titles(mid)
    seen = _first_last_seen(mid)
    excerpts = _top_chunks_per_cluster(
        mid, per_cluster=DEFAULT_CHUNKS_PER_CLUSTER, body_chars=DEFAULT_CHUNK_BODY_CHARS,
    )
    keyphrases = _cluster_keyphrases(mid)
    cluster_ids = sorted(
        clusters.cluster_keywords.keys() if clusters else [],
        key=lambda c: clusters.cluster_sizes.get(c, 0) if clusters else 0,
        reverse=True,
    )[:20]
    cluster_blocks = []
    excerpt_chars = 0
    for cid in cluster_ids:
        block = {
            "id": cid,
            "size": (clusters.cluster_sizes.get(cid, 0) if clusters else 0),
            "keywords": [w for w, _ in (clusters.cluster_keywords.get(cid, []) if clusters else [])[:8]],
            "keyphrases": keyphrases.get(cid, []),
            "representative_titles": titles.get(cid, []),
            "representative_quotes": quotes.get(cid, []),
            "excerpts": excerpts.get(cid, []),
            "first_seen": seen.get(cid, (None, None))[0].isoformat() if seen.get(cid, (None, None))[0] else "?",
            "last_seen": seen.get(cid, (None, None))[1].isoformat() if seen.get(cid, (None, None))[1] else "?",
        }
        cluster_blocks.append(block)
        for ex in excerpts.get(cid, []):
            excerpt_chars += len(ex.get("body", ""))
    # Plus the prompt template itself.
    from text_theme_analyzer.llm.prompts import SYSTEM_PROMPT
    prompt_overhead = len(SYSTEM_PROMPT) + 200  # schema + instructions
    total_chars = prompt_overhead + sum(
        len(__import__("json").dumps(b, default=str)) for b in cluster_blocks
    )
    return {
        "cluster_count": len(cluster_blocks),
        "chunk_count": len(chunks),
        "excerpt_chars": excerpt_chars,
        "est_tokens": total_chars // 4,
        "would_call_llm": True,
    }
