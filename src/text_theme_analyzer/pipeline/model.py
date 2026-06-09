"""Pipeline dataclasses shared across stages and output renderers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


@dataclass
class Note:
    """A single markdown note loaded from disk."""

    id: str
    path: Path
    title: str
    body: str
    date: date | None
    tags: list[str] = field(default_factory=list)
    frontmatter: dict[str, Any] = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.body.split())


@dataclass
class NoteChunk:
    """A chunk of a Note's body. Long notes are split into multiple chunks."""

    note_id: str
    chunk_index: int
    text: str
    char_offset: int


@dataclass
class ClusterResult:
    """Output of the clustering stage."""

    assignments: list[int]  # one per chunk, -1 = outlier
    cluster_sizes: dict[int, int]
    cluster_keywords: dict[int, list[tuple[str, float]]]  # c-TF-IDF
    cluster_representatives: dict[int, list[str]]  # note IDs closest to centroid
    umap_2d: list[tuple[float, float]]  # one per chunk
    outlier_count: int


@dataclass
class Spike:
    """A cluster with an unusually high count in a particular time bucket."""

    cluster_id: int
    bucket: date
    count: int
    rolling_mean: float
    delta: float  # count - rolling_mean


@dataclass
class StaleIdea:
    """A cluster that used to recur but has gone quiet in the recent window.

    `severity` is a coarse ladder:
    - "strong": a topic you wrote about often, then stopped (≥5 notes,
      last seen in the *first half* of the data range, silent in the
      recent `stale_window`).
    - "medium": recurred enough to notice (≥3 notes) but silent recently.
    - "weak": 2+ notes, silent for an extended window (2× the configured
      stale window).
    `quiet_streak_buckets` reports how many consecutive empty buckets
    preceded the latest data point — useful for the LLM verdict and the
    dashboard.
    """

    cluster_id: int
    first_seen: date
    last_seen: date
    frequency: int
    severity: str = "medium"
    quiet_streak_buckets: int = 0


@dataclass
class ThemeTimeseries:
    """Bucket-counted cluster activity over time."""

    bucket: str  # "week" or "month"
    series: dict[int, dict[date, int]]  # cluster_id -> {bucket_date: count}
    spikes: list[Spike] = field(default_factory=list)
    stale: list[StaleIdea] = field(default_factory=list)


@dataclass
class Analysis:
    """The full pipeline output. Every renderer consumes this."""

    notes: list[Note]
    chunks: list[NoteChunk]
    chunk_note_ids: list[str]  # parallel to chunks; assigns each chunk to its parent note
    keywords: dict[str, list[tuple[str, float]]]  # note_id -> [(phrase, score)]
    keyphrase_frequency: list[tuple[str, int]]  # corpus-level, sorted desc
    clusters: ClusterResult | None
    timeseries: ThemeTimeseries | None
    enrichment: Any | None = None  # llm.schemas.EnrichmentResult; kept as Any to avoid import cycle
    metadata: dict[str, Any] = field(default_factory=dict)
