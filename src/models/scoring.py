"""Pydantic models for scoring and ranking output."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class CandidateScore(BaseModel):
    """Intermediate scoring object for a candidate."""
    candidate_id: str
    semantic_score: float = Field(default=0.0, ge=0.0, le=1.0)
    signal_score: float = Field(default=0.0, ge=0.0, le=1.0)
    hybrid_score: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_score: Optional[float] = Field(default=None)
    final_score: float = Field(default=0.0)
    reasoning: str = Field(default="")

    # Breakdown components for transparency
    skill_match: float = Field(default=0.0)
    experience_score: float = Field(default=0.0)
    engagement_score: float = Field(default=0.0)
    reliability_score: float = Field(default=0.0)
    availability_score: float = Field(default=0.0)


class RankingResult(BaseModel):
    """Final ranked candidate for submission."""
    candidate_id: str
    rank: int
    score: float
    reasoning: str
