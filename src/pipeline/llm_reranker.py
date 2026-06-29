"""
LLM Reranker Node — uses Groq LLM to intelligently rerank the top-150
hybrid-scored candidates.

Strategy:
- Process candidates in batches of 10
- For each batch, ask the LLM to rank them 1-N with a numeric score and reason
- Aggregate all LLM scores and produce a final merged ranking
- Use llama-3.3-70b-versatile for quality, with structured output
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.models.candidate import Candidate
from src.models.job_description import ParsedJD
from src.models.scoring import CandidateScore
from src.utils.logging_utils import get_logger
from src.utils.text_utils import build_candidate_text

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a world-class technical recruiter with deep expertise in talent assessment.
You will be given a job description and a batch of candidate profiles.
Your task: rank these candidates from best to worst fit for the role.

For each candidate return a JSON array where each element has these exact keys:
- candidate_id: the CAND_XXXXXXX id string
- score: float 0.0-1.0 (1.0 = perfect fit)
- reasoning: 1 concise sentence explaining why this candidate fits or doesn't

Scoring guidelines:
- 0.85-1.0: Exceptional fit — directly relevant title, required skills, right experience
- 0.65-0.84: Strong fit — most requirements met, minor gaps
- 0.45-0.64: Moderate fit — relevant background but significant gaps
- 0.20-0.44: Weak fit — few relevant signals
- 0.00-0.19: Poor fit — mismatch in core requirements

Be a great recruiter: look beyond keywords. Consider career trajectory, consistency,
skill depth (not just presence), platform signals.

Return ONLY a JSON array. No markdown, no explanation outside the JSON.
Example format (use double curly braces are NOT needed — return real JSON):
[
  {{"candidate_id": "CAND_0000001", "score": 0.92, "reasoning": "Strong ML background with 6y production experience and relevant NLP skills."}},
  {{"candidate_id": "CAND_0000002", "score": 0.45, "reasoning": "Operations background with minimal AI/ML exposure."}}
]
"""

_HUMAN_PROMPT = """\
Job Description:
{jd_summary}

Candidates to rank (batch {batch_num}/{total_batches}):
{candidates_text}

Return the JSON array now.
"""


def _format_candidate_for_llm(candidate: Candidate, hybrid_score: CandidateScore) -> str:
    """Format a candidate profile concisely for the LLM prompt."""
    p = candidate.profile
    s = candidate.redrob_signals

    # Top skills
    top_skills = sorted(
        candidate.skills,
        key=lambda x: (
            {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}.get(x.proficiency, 0),
            x.endorsements,
        ),
        reverse=True,
    )[:8]
    skills_str = ", ".join(f"{sk.name} ({sk.proficiency})" for sk in top_skills)

    # Recent 2 roles
    roles = []
    for r in candidate.career_history[:2]:
        roles.append(f"  - {r.title} @ {r.company} ({r.duration_months}mo): {r.description[:120]}...")

    lines = [
        f"[{candidate.candidate_id}]",
        f"  Title: {p.current_title} | Exp: {p.years_of_experience}y | Industry: {p.current_industry}",
        f"  Headline: {p.headline}",
        f"  Skills: {skills_str}",
        f"  Career:",
    ] + roles + [
        f"  Platform: response_rate={s.recruiter_response_rate:.0%}, "
        f"interview_completion={s.interview_completion_rate:.0%}, "
        f"open_to_work={s.open_to_work_flag}, notice={s.notice_period_days}d",
        f"  Hybrid score (pre-LLM): {hybrid_score.hybrid_score:.3f}",
    ]
    return "\n".join(lines)


class LLMReranker:
    """
    Uses Groq LLM to rerank the top hybrid candidates with genuine reasoning.
    Works in batches to respect token limits.
    """

    def __init__(
        self,
        parsed_jd: ParsedJD,
        groq_api_key: str,
        batch_size: int = 10,
        max_retries: int = 3,
    ):
        self.jd = parsed_jd
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=3000,
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", _HUMAN_PROMPT),
        ])
        self.chain = self.prompt | self.llm | JsonOutputParser()

    def _jd_summary(self) -> str:
        jd = self.jd
        return (
            f"Role: {jd.role_title}\n"
            f"Summary: {jd.role_summary}\n"
            f"Required skills: {', '.join(jd.required_skills)}\n"
            f"Preferred skills: {', '.join(jd.preferred_skills)}\n"
            f"Experience: {jd.required_experience_years}+ years\n"
            f"Key responsibilities: {'; '.join(jd.key_responsibilities[:5])}\n"
            f"Disqualifiers: {'; '.join(jd.disqualifiers) if jd.disqualifiers else 'None'}"
        )

    def _call_llm_with_retry(self, payload: dict) -> List[Dict]:
        for attempt in range(self.max_retries):
            try:
                result = self.chain.invoke(payload)
                if isinstance(result, list):
                    return result
                # Sometimes the model wraps in a dict
                if isinstance(result, dict):
                    for v in result.values():
                        if isinstance(v, list):
                            return v
                logger.warning(f"Unexpected LLM output type: {type(result)}, attempt {attempt+1}")
            except Exception as e:
                raw_msg = str(e)
                # Try to extract JSON from the raw LLM text if parser failed
                extracted = self._try_extract_json(raw_msg)
                if extracted:
                    return extracted
                # Detect rate limit and wait appropriately
                if "rate_limit_exceeded" in raw_msg or "429" in raw_msg:
                    wait = self._parse_rate_limit_wait(raw_msg)
                    logger.warning(
                        f"Rate limit hit (attempt {attempt+1}/{self.max_retries}). "
                        f"Waiting {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    logger.warning(f"LLM call failed (attempt {attempt+1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
        return []

    @staticmethod
    def _parse_rate_limit_wait(msg: str) -> float:
        """Extract wait seconds from Groq rate limit error message."""
        import re
        # "Please try again in 34m27.552s"
        match = re.search(r'try again in\s+(\d+)m([\d.]+)s', msg)
        if match:
            minutes = int(match.group(1))
            seconds = float(match.group(2))
            total = minutes * 60 + seconds
            # Cap at 5 minutes to avoid hanging forever; warn if longer
            if total > 300:
                logger.warning(
                    f"Rate limit requires {total:.0f}s wait — capping at 300s. "
                    "Consider upgrading Groq tier or rerunning after the quota resets."
                )
                return 300.0
            return total + 2  # small buffer
        # "try again in 45.2s"
        match2 = re.search(r'try again in\s+([\d.]+)s', msg)
        if match2:
            return min(300.0, float(match2.group(1)) + 2)
        return 60.0  # default fallback

    @staticmethod
    def _try_extract_json(text: str) -> List[Dict]:
        """Try to extract a JSON array from error/raw text using regex fallback."""
        import re, json
        # Find the first [...] block in the text
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if not match:
            return []
        raw = match.group(0)
        # Fix trailing commas before ] or }
        raw = re.sub(r',\s*([\]}])', r'\1', raw)
        try:
            return json.loads(raw)
        except Exception:
            return []

    def rerank(
        self,
        candidates: List[Candidate],
        hybrid_scores: List[CandidateScore],
        top_n: int = 100,
    ) -> List[CandidateScore]:
        """
        Rerank using LLM. Returns top_n CandidateScore objects with LLM scores and reasoning.
        candidates and hybrid_scores must be aligned (same order).
        """
        assert len(candidates) == len(hybrid_scores)

        jd_summary = self._jd_summary()
        id_to_hybrid = {hs.candidate_id: hs for hs in hybrid_scores}
        id_to_candidate = {c.candidate_id: c for c in candidates}

        # Store LLM scores per candidate_id
        llm_scores: Dict[str, float] = {}
        llm_reasoning: Dict[str, str] = {}

        total_batches = (len(candidates) + self.batch_size - 1) // self.batch_size
        logger.info(f"LLM reranking {len(candidates)} candidates in {total_batches} batches...")

        for batch_idx in range(total_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(candidates))
            batch_candidates = candidates[start:end]

            candidates_text = "\n\n".join(
                _format_candidate_for_llm(c, id_to_hybrid[c.candidate_id])
                for c in batch_candidates
            )

            payload = {
                "jd_summary": jd_summary,
                "candidates_text": candidates_text,
                "batch_num": batch_idx + 1,
                "total_batches": total_batches,
            }

            batch_results = self._call_llm_with_retry(payload)

            for item in batch_results:
                cid = item.get("candidate_id", "").strip()
                score = item.get("score", None)
                reasoning = item.get("reasoning", "")
                if cid and score is not None:
                    try:
                        llm_scores[cid] = float(score)
                        llm_reasoning[cid] = str(reasoning)
                    except (ValueError, TypeError):
                        pass

            logger.info(
                f"  Batch {batch_idx+1}/{total_batches} done — "
                f"{len(batch_results)} scored"
            )
            # Rate limit: small pause between batches
            if batch_idx < total_batches - 1:
                time.sleep(0.3)

        # Combine: LLM score (60%) + hybrid score (40%) for final ranking
        final_scores: List[CandidateScore] = []
        for cid, hybrid_score in id_to_hybrid.items():
            llm_s = llm_scores.get(cid)
            hybrid_s = hybrid_score.hybrid_score

            if llm_s is not None:
                final_s = 0.60 * llm_s + 0.40 * hybrid_s
                reasoning = llm_reasoning.get(cid, "Ranked by hybrid scoring.")
            else:
                # LLM didn't score this candidate — fall back to hybrid only
                final_s = hybrid_s * 0.90  # small penalty for no LLM signal
                reasoning = (
                    f"{hybrid_score.experience_score and 'Experience match. ' or ''}"
                    f"Skill match: {hybrid_score.skill_match:.0%}. "
                    f"Signal score: {hybrid_score.signal_score:.0%}."
                )

            updated = hybrid_score.model_copy(update={
                "llm_score": llm_s,
                "final_score": round(min(1.0, max(0.0, final_s)), 4),
                "reasoning": reasoning,
            })
            final_scores.append(updated)

        # Sort by final score descending
        final_scores.sort(key=lambda x: x.final_score, reverse=True)

        logger.info(
            f"LLM reranking complete — "
            f"top-{min(top_n, len(final_scores))} produced"
        )
        return final_scores[:top_n]
