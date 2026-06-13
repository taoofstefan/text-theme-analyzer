"""Pydantic schemas for the LLM enrichment response.

The LLM receives a bundle of cluster metadata and returns a single JSON
object matching `EnrichmentResult`. We validate strictly; any mismatch
triggers one retry with the error appended to the prompt.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClusterAnnotation(BaseModel):
    cluster_id: int
    # Optional: when the LLM returns a name, we use it; otherwise the
    # renderer falls back to a deterministic "{kw1} / {kw2}" name from the
    # cluster's top c-TF-IDF keywords.
    name: str | None = Field(default=None, max_length=160)
    summary: str = Field(..., min_length=5, max_length=1000)
    top_quotes: list[str] = Field(default_factory=list, max_length=5)
    emotional_tone: str = Field(..., min_length=2, max_length=120)


class Tension(BaseModel):
    title: str = Field(..., min_length=2, max_length=80)
    pole_a: str = Field(..., min_length=2, max_length=200)
    pole_b: str = Field(..., min_length=2, max_length=200)
    evidence: list[str] = Field(default_factory=list, max_length=5)
    note: str = Field(..., min_length=2, max_length=400)


class ArticleCandidate(BaseModel):
    title: str = Field(..., min_length=2, max_length=120)
    angle: str = Field(..., min_length=5, max_length=400)
    supporting_cluster_ids: list[int] = Field(default_factory=list)


class StaleVerdict(BaseModel):
    cluster_id: int
    theme: str = Field(..., min_length=2, max_length=120)
    verdict: Literal["promote_to_project", "archive", "keep_observing"]
    reasoning: str = Field(..., min_length=5, max_length=400)
    target_section: str | None = Field(default=None, max_length=80)


class QuoteValidation(BaseModel):
    """How many of the LLM's returned quotes didn't appear verbatim in the input."""

    requested: int = 0
    dropped: int = 0


class EnrichmentResult(BaseModel):
    clusters: list[ClusterAnnotation] = Field(default_factory=list)
    tensions: list[Tension] = Field(default_factory=list)
    article_candidates: list[ArticleCandidate] = Field(default_factory=list)
    stale_recurring: list[StaleVerdict] = Field(default_factory=list)
    quote_validation: QuoteValidation = Field(default_factory=QuoteValidation)
