"""Text utilities: build rich text representations for embedding."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.candidate import Candidate
    from src.models.job_description import ParsedJD


PROFICIENCY_WEIGHT = {
    "beginner": 0.25,
    "intermediate": 0.5,
    "advanced": 0.8,
    "expert": 1.0,
}


def build_candidate_text(candidate: "Candidate") -> str:
    """
    Build a rich, information-dense text string for embedding a candidate.
    Captures title, summary, skills, career history, and education.
    """
    p = candidate.profile
    parts: list[str] = []

    # Header
    parts.append(f"Title: {p.current_title}")
    parts.append(f"Headline: {p.headline}")
    parts.append(f"Experience: {p.years_of_experience} years")
    parts.append(f"Industry: {p.current_industry}")
    parts.append(f"Summary: {p.summary[:400]}")  # keep it concise

    # Skills (emphasise advanced/expert ones)
    high_skills = [
        s.name for s in candidate.skills
        if s.proficiency in ("advanced", "expert")
    ]
    mid_skills = [
        s.name for s in candidate.skills
        if s.proficiency == "intermediate"
    ]
    if high_skills:
        parts.append(f"Expert skills: {', '.join(high_skills)}")
    if mid_skills:
        parts.append(f"Intermediate skills: {', '.join(mid_skills[:10])}")

    # Career (last 3 roles)
    for role in candidate.career_history[:3]:
        parts.append(
            f"Role: {role.title} at {role.company} ({role.industry}) "
            f"for {role.duration_months} months. {role.description[:200]}"
        )

    # Education
    for edu in candidate.education[:2]:
        parts.append(f"Education: {edu.degree} in {edu.field_of_study} from {edu.institution} ({edu.tier})")

    # Certifications
    certs = [c.name for c in candidate.certifications[:3]]
    if certs:
        parts.append(f"Certifications: {', '.join(certs)}")

    return " | ".join(parts)


def build_jd_text(parsed_jd: "ParsedJD") -> str:
    """Build embedding text from a parsed JD."""
    return parsed_jd.build_embedding_text()
