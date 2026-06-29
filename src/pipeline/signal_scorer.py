"""
Signal Scorer Node — scores candidates on Redrob platform signals.

Signals scored:
1. Engagement: recruiter_response_rate, avg_response_time_hours
2. Reliability: interview_completion_rate, offer_acceptance_rate
3. Availability: open_to_work_flag, notice_period_days
4. Profile quality: profile_completeness_score, verified contacts
5. Activity: last_active_date, search_appearance_30d, saved_by_recruiters_30d
6. Skill validation: skill_assessment_scores, github_activity_score
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import List, Tuple

from src.models.candidate import Candidate
from src.models.job_description import ParsedJD
from src.models.scoring import CandidateScore
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Weights for each signal component
SIGNAL_WEIGHTS = {
    "engagement": 0.25,      # responsiveness to recruiters
    "reliability": 0.20,     # interview/offer follow-through
    "availability": 0.15,    # open to work, short notice period
    "profile_quality": 0.15, # completeness, verified contacts
    "activity": 0.15,        # recently active, being found
    "skill_validation": 0.10,# assessment scores, github
}

TODAY = date.today()


def _days_since(date_str: str) -> int:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return max(0, (TODAY - d).days)
    except Exception:
        return 365


def _sigmoid(x: float) -> float:
    """Smooth bounded sigmoid for continuous scores."""
    return 1.0 / (1.0 + math.exp(-x))


class SignalScorer:
    """
    Compute a normalized signal score (0-1) for each candidate based on
    their Redrob platform behavioral signals.
    """

    def __init__(self, parsed_jd: ParsedJD):
        self.jd = parsed_jd

    def score(self, candidate: Candidate) -> float:
        """Return overall signal score 0-1."""
        s = candidate.redrob_signals
        components = {
            "engagement": self._engagement(s),
            "reliability": self._reliability(s),
            "availability": self._availability(s),
            "profile_quality": self._profile_quality(s),
            "activity": self._activity(s),
            "skill_validation": self._skill_validation(s, candidate),
        }
        total = sum(SIGNAL_WEIGHTS[k] * v for k, v in components.items())
        return round(min(1.0, max(0.0, total)), 4)

    def score_batch(
        self,
        candidates: List[Candidate],
    ) -> List[Tuple[Candidate, float]]:
        results = [(c, self.score(c)) for c in candidates]
        logger.info(f"Signal scores computed for {len(results)} candidates")
        return results

    # ── Component scorers ──────────────────────────────────────────────────

    def _engagement(self, s) -> float:
        """How responsive is the candidate to recruiters?"""
        response_score = s.recruiter_response_rate  # 0-1 directly

        # Lower response time is better; normalise: 0h → 1.0, 168h (7 days) → 0
        time_score = max(0.0, 1.0 - s.avg_response_time_hours / 168.0)

        return 0.6 * response_score + 0.4 * time_score

    def _reliability(self, s) -> float:
        """Does the candidate actually show up and accept offers?"""
        interview_score = s.interview_completion_rate  # 0-1

        # offer_acceptance_rate can be -1 (no history) → treat as neutral 0.5
        if s.offer_acceptance_rate < 0:
            offer_score = 0.5
        else:
            offer_score = s.offer_acceptance_rate

        return 0.6 * interview_score + 0.4 * offer_score

    def _availability(self, s) -> float:
        """Is the candidate actually looking and available quickly?"""
        open_score = 1.0 if s.open_to_work_flag else 0.3

        # Notice period: 0-30 days → 1.0, 31-60 → 0.7, 61-90 → 0.4, >90 → 0.1
        if s.notice_period_days <= 30:
            notice_score = 1.0
        elif s.notice_period_days <= 60:
            notice_score = 0.7
        elif s.notice_period_days <= 90:
            notice_score = 0.4
        else:
            notice_score = 0.1

        return 0.5 * open_score + 0.5 * notice_score

    def _profile_quality(self, s) -> float:
        """Is the profile complete and verified?"""
        completeness = s.profile_completeness_score / 100.0
        verifications = sum([
            s.verified_email,
            s.verified_phone,
            s.linkedin_connected,
        ]) / 3.0
        return 0.6 * completeness + 0.4 * verifications

    def _activity(self, s) -> float:
        """Is the candidate recently active and visible?"""
        # Recency: 0-30 days → 1.0, drops off
        days = _days_since(s.last_active_date)
        recency = max(0.0, 1.0 - days / 180.0)

        # Search appearances: log-scaled, 0 → 0, 100 → ~0.5, 500+ → ~1.0
        search_score = min(1.0, math.log1p(s.search_appearance_30d) / math.log1p(500))

        # Saved by recruiters (demand signal)
        saved_score = min(1.0, s.saved_by_recruiters_30d / 10.0)

        return 0.4 * recency + 0.35 * search_score + 0.25 * saved_score

    def _skill_validation(self, s, candidate: Candidate) -> float:
        """Validated skill depth: assessment scores + github activity."""
        assessment_scores = list(s.skill_assessment_scores.values())
        if assessment_scores:
            avg_assessment = sum(assessment_scores) / len(assessment_scores) / 100.0
        else:
            avg_assessment = 0.4  # neutral prior — no data

        # GitHub: -1 means not linked; 0-100 otherwise
        if s.github_activity_score >= 0:
            github_score = s.github_activity_score / 100.0
        else:
            github_score = 0.3  # not linked → slight negative signal

        # Endorsements from peers (log-scaled)
        endorsement_score = min(1.0, math.log1p(s.endorsements_received) / math.log1p(100))

        return 0.4 * avg_assessment + 0.35 * github_score + 0.25 * endorsement_score
