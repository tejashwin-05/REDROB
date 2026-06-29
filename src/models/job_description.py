"""Pydantic models for job description and parsed requirements."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class JobDescription(BaseModel):
    """Raw job description text."""
    raw_text: str
    source_file: Optional[str] = None


class ParsedJD(BaseModel):
    """Structured requirements extracted from a job description by the LLM."""
    role_title: str = Field(description="The primary role title being hired for")
    role_summary: str = Field(description="1-2 sentence summary of the role")
    required_skills: List[str] = Field(description="Core technical/functional skills required")
    preferred_skills: List[str] = Field(description="Nice-to-have skills")
    required_experience_years: float = Field(description="Minimum years of relevant experience")
    max_experience_years: Optional[float] = Field(default=None, description="Maximum experience (if mentioned)")
    industries_preferred: List[str] = Field(default_factory=list, description="Preferred industry backgrounds")
    education_requirements: str = Field(description="Education requirements (degree level, field)")
    key_responsibilities: List[str] = Field(description="Top 5 key responsibilities")
    disqualifiers: List[str] = Field(default_factory=list, description="Hard disqualifying factors")
    work_mode: Optional[str] = Field(default=None, description="remote/hybrid/onsite/flexible")
    location_preference: Optional[str] = Field(default=None, description="Preferred location or 'any'")
    embedding_text: str = Field(default="", description="Flattened text for embedding the JD")

    def build_embedding_text(self) -> str:
        """Create a rich text representation for semantic embedding."""
        parts = [
            f"Role: {self.role_title}",
            f"Summary: {self.role_summary}",
            f"Required skills: {', '.join(self.required_skills)}",
            f"Preferred skills: {', '.join(self.preferred_skills)}",
            f"Experience: {self.required_experience_years}+ years",
            f"Responsibilities: {' | '.join(self.key_responsibilities)}",
        ]
        if self.industries_preferred:
            parts.append(f"Industries: {', '.join(self.industries_preferred)}")
        return " | ".join(parts)
