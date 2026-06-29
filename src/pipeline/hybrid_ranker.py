"""
Hybrid Ranker Node — combines semantic similarity + signal scores + skill match
into a single hybrid score to select the top-N candidates for LLM reranking.

Scoring formula (configurable):
  hybrid_score = w_sem * semantic + w_sig * signal + w_skill * skill_match

Then an experience gate is applied: candidates outside the required experience
range get a penalty.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from rapidfuzz import fuzz

from src.models.candidate import Candidate
from src.models.job_description import ParsedJD
from src.models.scoring import CandidateScore
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Default hybrid weights
W_SEMANTIC = 0.45
W_SIGNAL = 0.30
W_SKILL = 0.25

PROFICIENCY_SCORE = {
    "beginner": 0.25,
    "intermediate": 0.55,
    "advanced": 0.80,
    "expert": 1.00,
}


class HybridRanker:
    """
    Combines semantic, signal, and skill-match scores into a unified hybrid score.
    Also applies soft experience gating.
    """

    def __init__(
        self,
        parsed_jd: ParsedJD,
        w_semantic: float = W_SEMANTIC,
        w_signal: float = W_SIGNAL,
        w_skill: float = W_SKILL,
    ):
        self.jd = parsed_jd
        self.w_semantic = w_semantic
        self.w_signal = w_signal
        self.w_skill = w_skill
        self._required_skills_lower = [s.lower() for s in parsed_jd.required_skills]
        self._preferred_skills_lower = [s.lower() for s in parsed_jd.preferred_skills]

    def compute_skill_match(self, candidate: Candidate) -> float:
        """
        Compute a skill match score against the JD's required and preferred skills.
        Uses fuzzy matching to handle spelling variations.
        """
        if not self._required_skills_lower:
            return 0.5  # no required skills defined → neutral

        candidate_skills = {
            s.name.lower(): (s.proficiency, s.endorsements, s.duration_months or 0)
            for s in candidate.skills
        }

        # Required skills match (more important)
        req_match_scores = []
        for req in self._required_skills_lower:
            best = 0.0
            for cand_skill, (prof, endorsements, duration) in candidate_skills.items():
                similarity = fuzz.token_sort_ratio(req, cand_skill) / 100.0
                if similarity >= 0.7:
                    proficiency_weight = PROFICIENCY_SCORE.get(prof, 0.4)
                    # Bonus for long usage duration (capped at 36 months)
                    duration_bonus = min(1.0, duration / 36.0) * 0.15
                    # Endorsement bonus (capped at 20)
                    endorse_bonus = min(1.0, endorsements / 20.0) * 0.10
                    match = similarity * proficiency_weight + duration_bonus + endorse_bonus
                    best = max(best, min(1.0, match))
            req_match_scores.append(best)

        required_score = sum(req_match_scores) / len(req_match_scores) if req_match_scores else 0.0

        # Preferred skills match (less important)
        if self._preferred_skills_lower:
            pref_matches = []
            for pref in self._preferred_skills_lower:
                best = 0.0
                for cand_skill in candidate_skills:
                    sim = fuzz.token_sort_ratio(pref, cand_skill) / 100.0
                    if sim >= 0.7:
                        best = max(best, sim * PROFICIENCY_SCORE.get(
                            candidate_skills[cand_skill][0], 0.4
                        ))
                pref_matches.append(best)
            preferred_score = sum(pref_matches) / len(pref_matches)
        else:
            preferred_score = 0.0

        # Combined: required is 75%, preferred is 25%
        return 0.75 * required_score + 0.25 * preferred_score

    def experience_gate(self, candidate: Candidate) -> float:
        """
        Soft experience gate: returns a multiplier [0.3, 1.0].
        Penalises candidates too junior or too senior for the role.
        """
        yoe = candidate.profile.years_of_experience
        min_exp = self.jd.required_experience_years
        max_exp = self.jd.max_experience_years

        if yoe < min_exp:
            deficit = min_exp - yoe
            # Graceful degradation: 1 year under → 0.85, 3+ years under → 0.3
            return max(0.30, 1.0 - deficit * 0.15)
        elif max_exp and yoe > max_exp:
            excess = yoe - max_exp
            return max(0.60, 1.0 - excess * 0.08)  # overqualification soft penalty
        else:
            return 1.0

    def score(
        self,
        candidate: Candidate,
        semantic_score: float,
        signal_score: float,
    ) -> CandidateScore:
        """Compute the full hybrid score for a candidate."""
        skill_match = self.compute_skill_match(candidate)
        exp_gate = self.experience_gate(candidate)

        raw_hybrid = (
            self.w_semantic * semantic_score
            + self.w_signal * signal_score
            + self.w_skill * skill_match
        )
        hybrid_score = raw_hybrid * exp_gate

        return CandidateScore(
            candidate_id=candidate.candidate_id,
            semantic_score=round(semantic_score, 4),
            signal_score=round(signal_score, 4),
            skill_match=round(skill_match, 4),
            hybrid_score=round(min(1.0, hybrid_score), 4),
            experience_score=round(exp_gate, 4),
            final_score=round(min(1.0, hybrid_score), 4),
        )

    def rank(
        self,
        candidates_with_scores: List[Tuple[Candidate, float, float]],
        top_n: int = 150,
    ) -> List[CandidateScore]:
        """
        Rank candidates by hybrid score.
        candidates_with_scores: list of (Candidate, semantic_score, signal_score)
        Returns top_n CandidateScore objects sorted by hybrid_score desc.
        """
        scored = [
            self.score(c, sem, sig)
            for c, sem, sig in candidates_with_scores
        ]
        scored.sort(key=lambda x: x.hybrid_score, reverse=True)
        logger.info(
            f"Hybrid ranking complete — "
            f"top-{min(top_n, len(scored))} selected from {len(scored)} candidates"
        )
        return scored[:top_n]
