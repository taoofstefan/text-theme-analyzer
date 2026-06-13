"""Keyphrase extraction with KeyBERT, YAKE as zero-dep fallback."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from math import log

from text_theme_analyzer.pipeline.model import Note, NoteChunk

# Alphabetic tokens only. Hyphens and digits are stripped so phrases like
# "state-of-the-art" collapse to usable words; the downstream aggregator
# already treats short tokens and pure stopwords as noise.
_TOKEN_RE = re.compile(r"[a-zA-Z]+")

# Common stopwords used to filter phrases that are all stopwords.
# KeyBERT already drops these, but YAKE doesn't, and chunk-level rollups
# sometimes still surface them.
STOPWORDS: frozenset[str] = frozenset([
    "a", "an", "the", "and", "or", "but", "if", "of", "in", "on", "at", "to",
    "for", "with", "by", "from", "as", "is", "was", "are", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "should", "could", "may", "might", "must", "shall", "can", "need",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "we", "our", "you", "your", "i", "my", "me", "he", "she", "his", "her",
    "not", "no", "yes", "so", "just", "also", "very", "more", "most", "some",
    "any", "all", "each", "every", "other", "another", "such", "same",
    "than", "then", "there", "here", "when", "where", "why", "how", "what",
    "which", "who", "whom", "whose",
])


def _is_mostly_stopwords(phrase: str) -> bool:
    """Return True if every word in the phrase is a stopword."""
    words = phrase.lower().split()
    return all(w in STOPWORDS for w in words) if words else True


def _dedupe_contained_phrases(
    counts: list[tuple[str, int]],
    *,
    enabled: bool = True,
    ratio: float = 1.5,
) -> list[tuple[str, int]]:
    """Drop a phrase when a strict-prefix counterpart has comparable frequency.

    For two phrases A (shorter) and B (longer) where A's words are a strict
    word-boundary prefix of B's words, if their counts are within a factor
    of `ratio` of each other, drop the shorter (the longer is the more
    specific phrasing at the same information level).

    - "discord direct" (5) vs "discord direct conversation" (5): ratio 1.0
      -> drop "discord direct" (the longer is the more specific signal).
    - "signal" (5) vs "signal lost" (2): ratio 2.5, exceeds the 1.5 cap
      -> keep both (the broader concept and the specific case are both
      informative).
    - "signal" (5) vs "signal scouts" (5): no prefix relation, not affected.

    Rationale: when counts are within ratio, the longer phrase is carrying
    all the information of the shorter plus more — strictly more useful.
    When counts diverge, the broader concept is doing work the longer isn't.
    """
    if not enabled:
        return counts

    # Tokenize once.
    tokenized: list[tuple[str, int, list[str]]] = []
    for phrase, count in counts:
        words = phrase.lower().split()
        if words:
            tokenized.append((phrase, count, words))

    # Iterate by words-length asc so shorter phrases are processed first.
    tokenized.sort(key=lambda t: (len(t[2]), -t[1]))

    dominated: set[str] = set()
    for i, (phrase_a, count_a, words_a) in enumerate(tokenized):
        if phrase_a in dominated:
            continue
        # Look for any B with A as a strict prefix of B's words.
        for j in range(i + 1, len(tokenized)):
            phrase_b, count_b, words_b = tokenized[j]
            if phrase_b in dominated:
                continue
            if words_b[: len(words_a)] != words_a:
                continue
            # Counts are within ratio of each other -> drop the shorter.
            lo, hi = sorted((count_a, count_b))
            if lo == 0 or hi / lo <= ratio:
                dominated.add(phrase_a)
                break
    keep = [(p, c) for p, c in counts if p not in dominated]
    return keep


def _aggregate_frequency(
    per_note: dict[str, list[tuple[str, float]]],
    *,
    merge_contained_phrases: bool = True,
) -> list[tuple[str, int]]:
    """Count how many notes each keyphrase appears in (document frequency).

    Post-processing:
    - Drop single-character phrases ("a", "I")
    - Drop phrases made entirely of stopwords
    - Optionally drop short phrases that are strict prefixes of longer
      phrases with equal-or-greater frequency.
    """
    counts: Counter[str] = Counter()
    for phrases in per_note.values():
        seen: set[str] = set()
        for phrase, _score in phrases:
            key = phrase.lower().strip()
            if len(key) < 2:
                continue
            if _is_mostly_stopwords(key):
                continue
            if key in seen:
                continue
            seen.add(key)
            counts[key] += 1
    ranked = counts.most_common()
    if merge_contained_phrases:
        ranked = _dedupe_contained_phrases(ranked, enabled=True)
    return ranked


def extract_with_keybert(
    texts: list[str],
    *,
    top_n: int = 10,
    embedding_model: str | None = None,
) -> list[list[tuple[str, float]]]:
    """Run KeyBERT on each text. Returns per-text list of (phrase, score)."""
    from keybert import KeyBERT

    kw_model = KeyBERT(model=embedding_model) if embedding_model else KeyBERT()
    out: list[list[tuple[str, float]]] = []
    for text in texts:
        if not text.strip():
            out.append([])
            continue
        try:
            kws = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words="english",
                top_n=top_n,
                use_mmr=True,
                diversity=0.5,
            )
        except Exception:
            kws = []
        out.append([(str(p), float(s)) for p, s in kws])
    return out


def _tokens(text: str) -> list[str]:
    """Lowercase alphabetic tokens, dropping stopwords and single letters."""
    return [
        t.lower()
        for t in _TOKEN_RE.findall(text)
        if len(t) > 1 and t.lower() not in STOPWORDS
    ]


def _tfidf_lite_score(
    term: str,
    tf: Counter[str],
    doc_freq: Counter[str],
    n_docs: int,
) -> float:
    df = doc_freq.get(term, 1)
    idf = log(n_docs / max(df, 1)) + 1.0
    return log(1 + tf.get(term, 0)) * idf


def extract_zero_dep(
    texts: list[str],
    *,
    top_n: int = 10,
) -> list[list[tuple[str, float]]]:
    """Pure-Python keyword fallback: no external deps, lower quality than KeyBERT.

    Uses a TF-IDF-lite score over unigrams and simple contiguous n-grams
    (up to 3 words) of non-stopword tokens. The result shape mirrors
    KeyBERT/YAKE: a per-text list of (phrase, score) with higher scores
    for phrases that are frequent in the document and rare in the corpus.
    """
    tokenized = [_tokens(t) for t in texts]
    n_docs = max(len(texts), 1)

    # Document frequency over the whole batch.
    doc_freq: Counter[str] = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))

    out: list[list[tuple[str, float]]] = []
    for tokens in tokenized:
        if not tokens:
            out.append([])
            continue

        tf: Counter[str] = Counter(tokens)
        candidates: dict[str, float] = {}
        length = len(tokens)
        for i in range(length):
            for j in range(i + 1, min(i + 4, length + 1)):
                phrase = " ".join(tokens[i:j])
                if phrase in candidates:
                    continue
                phrase_terms = tokens[i:j]
                # Average per-word score keeps short and long phrases on a
                # comparable scale; the downstream aggregator still prefers
                # phrases that appear in many notes.
                candidates[phrase] = sum(
                    _tfidf_lite_score(t, tf, doc_freq, n_docs) for t in phrase_terms
                ) / len(phrase_terms)

        ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
        out.append(ranked[:top_n])
    return out


def extract_keyphrases(
    chunks: Iterable[NoteChunk],
    *,
    method: str = "keybert",
    top_n: int = 10,
    embedding_model: str | None = None,
) -> list[list[tuple[str, float]]]:
    """Extract keyphrases for each chunk. Result is parallel to `chunks`.

    `method="keybert"` is the default. If KeyBERT is not installed,
    it falls back to the pure-Python zero-dep extractor. The legacy
    `method="yake"` name is still accepted and also routes to the
    zero-dep extractor (the yake package is no longer imported).
    """
    chunk_list = list(chunks)
    texts = [c.text for c in chunk_list]
    if method == "keybert":
        try:
            return extract_with_keybert(texts, top_n=top_n, embedding_model=embedding_model)
        except ImportError:
            return extract_zero_dep(texts, top_n=top_n)
    if method == "yake":
        return extract_zero_dep(texts, top_n=top_n)
    raise ValueError(f"Unknown keyphrase method: {method}")


def keyphrases_for_notes(
    notes: list[Note],
    chunks_by_note: dict[str, list[NoteChunk]],
    per_chunk_global: list[list[tuple[str, float]]],
) -> dict[str, list[tuple[str, float]]]:
    """Roll chunk-level keyphrases up to a single list per note (best score per phrase).

    `per_chunk_global` is the flat list parallel to a global chunk iteration order
    that matches the order chunks are appended to `chunks_by_note` (note-by-note).
    """
    out: dict[str, list[tuple[str, float]]] = {}
    cursor = 0
    for note in notes:
        note_chunks = chunks_by_note.get(note.id, [])
        n = len(note_chunks)
        score_by_phrase: dict[str, float] = {}
        for phrases in per_chunk_global[cursor:cursor + n]:
            for phrase, score in phrases:
                key = phrase.lower().strip()
                score_by_phrase[key] = max(score_by_phrase.get(key, 0.0), score)
        cursor += n
        out[note.id] = sorted(
            score_by_phrase.items(), key=lambda kv: kv[1], reverse=True
        )
    return out
