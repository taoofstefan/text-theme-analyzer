"""Lightweight emotional-tone scoring using a small English affect lexicon.

This is the M5 stretch goal: a deterministic, no-LLM signal that gives the
"emotional tone over time" output mentioned in idea.txt. The LLM enrichment
can refine this in M3's `ClusterAnnotation.emotional_tone` field, but this
lexicon approach is fast, free, and good enough for a quick trend line.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


# Compact English affect word list (positive / negative / arousal markers).
# Deliberately small — this is a heuristic, not a model. Words are lowercase.
POSITIVE = frozenset([
    "good", "great", "love", "best", "happy", "enjoy", "excited", "wonderful",
    "beautiful", "amazing", "perfect", "thanks", "glad", "hope", "win",
    "trust", "calm", "fun", "energy", "energized", "curious", "delight",
    "satisfy", "satisfied", "reward", "win", "growth", "flow", "groove",
    "finished", "ship", "shipped",
])
NEGATIVE = frozenset([
    "bad", "hate", "worst", "sad", "angry", "tired", "frustrated", "annoyed",
    "broken", "fail", "failed", "fear", "scared", "anxious", "stressed",
    "wrong", "lost", "stuck", "dead", "wrong", "pain", "hard", "difficult",
    "struggle", "burnout", "exhausted", "lonely", "isolated", "doubt",
    "worry", "scary", "terrible", "horrible", "sucks",
])
HIGH_AROUSAL = frozenset([
    "fire", "fast", "intense", "wild", "crazy", "loud", "huge", "massive",
    "explode", "exploding", "extreme", "urgent", "spike", "surge",
])
LOW_AROUSAL = frozenset([
    "quiet", "slow", "calm", "soft", "patient", "boring", "still", "subtle",
    "tired", "sleep", "sleepy", "gentle",
])

_WORD_RE = re.compile(r"\b[a-z']+\b")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


@dataclass
class ToneScore:
    valence: float  # -1 (negative) .. +1 (positive)
    arousal: float  # -1 (low)     .. +1 (high)
    counts: dict[str, int]


def score_text(text: str) -> ToneScore:
    """Score a single chunk of text on valence + arousal dimensions."""
    tokens = _tokenize(text)
    n = max(1, len(tokens))
    pos = sum(1 for t in tokens if t in POSITIVE)
    neg = sum(1 for t in tokens if t in NEGATIVE)
    high = sum(1 for t in tokens if t in HIGH_AROUSAL)
    low = sum(1 for t in tokens if t in LOW_AROUSAL)
    valence = (pos - neg) / n
    arousal = (high - low) / n
    return ToneScore(
        valence=valence,
        arousal=arousal,
        counts={"positive": pos, "negative": neg, "high": high, "low": low},
    )


def aggregate_scores(per_note: dict[str, str]) -> dict[str, ToneScore]:
    """Score a dict of note_id -> body."""
    return {nid: score_text(body) for nid, body in per_note.items()}


def tone_label(score: ToneScore) -> str:
    """Return a short hyphenated label, e.g. 'frustrated-energized'."""
    val = (
        "negative" if score.valence < -0.005
        else "positive" if score.valence > 0.005
        else "neutral"
    )
    ar = (
        "high" if score.arousal > 0.005
        else "low" if score.arousal < -0.005
        else "calm"
    )
    return f"{val}-{ar}"


def tone_over_time(
    per_note: dict[str, str],
    note_dates: dict[str, "date | None"],
    *,
    bucket: str = "month",
) -> list[dict]:
    """Bucket scores per month and return a list of {bucket, valence, arousal, count}."""
    from text_theme_analyzer.pipeline.timeseries import _bucket_start  # type: ignore

    buckets: dict = {}
    for nid, body in per_note.items():
        d = note_dates.get(nid)
        if d is None:
            continue
        b = _bucket_start(d, bucket)
        s = score_text(body)
        slot = buckets.setdefault(b, {"valence_sum": 0.0, "arousal_sum": 0.0, "count": 0})
        slot["valence_sum"] += s.valence
        slot["arousal_sum"] += s.arousal
        slot["count"] += 1
    out = []
    for b in sorted(buckets):
        slot = buckets[b]
        n = slot["count"]
        out.append({
            "bucket": b.isoformat(),
            "valence": slot["valence_sum"] / n,
            "arousal": slot["arousal_sum"] / n,
            "count": n,
        })
    return out
