"""
Output Writer — generates the final submission CSV and metadata YAML.
Validates the output against challenge rules before writing.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import List

from src.models.scoring import CandidateScore, RankingResult
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

CANDIDATE_ID_RE = re.compile(r"^CAND_[0-9]{7}$")


def write_submission(
    ranked_scores: List[CandidateScore],
    output_path: str | Path,
    top_n: int = 100,
) -> Path:
    """
    Write the final submission CSV.

    Rules (from validate_submission.py):
    - Row 1: header = candidate_id,rank,score,reasoning
    - Rows 2-101: exactly 100 data rows
    - Ranks 1-100, unique, sequential
    - Scores non-increasing by rank
    - Tie-break: candidate_id ascending
    - candidate_id format: CAND_XXXXXXX (7 digits)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure we have exactly top_n candidates
    candidates = ranked_scores[:top_n]
    if len(candidates) < top_n:
        logger.warning(
            f"Only {len(candidates)} candidates available; submission will have {len(candidates)} rows "
            f"(need {top_n} for valid submission)."
        )

    # Sort by final_score desc, break ties by candidate_id asc
    candidates = sorted(
        candidates,
        key=lambda x: (-x.final_score, x.candidate_id),
    )

    # Ensure scores are non-increasing (fix any floating point issues)
    for i in range(1, len(candidates)):
        if candidates[i].final_score > candidates[i - 1].final_score:
            candidates[i] = candidates[i].model_copy(
                update={"final_score": candidates[i - 1].final_score}
            )

    results: List[RankingResult] = []
    for rank, cs in enumerate(candidates, start=1):
        # Sanitise reasoning: no newlines, max 300 chars
        reason = cs.reasoning.replace("\n", " ").replace(",", ";").strip()
        reason = reason[:300] if len(reason) > 300 else reason

        results.append(RankingResult(
            candidate_id=cs.candidate_id,
            rank=rank,
            score=round(cs.final_score, 4),
            reasoning=reason,
        ))

    # Validate before writing
    _validate_results(results)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in results:
            writer.writerow([r.candidate_id, r.rank, r.score, r.reasoning])

    logger.info(f"Submission written → {output_path} ({len(results)} rows)")
    return output_path


def _validate_results(results: List[RankingResult]) -> None:
    """Raise ValueError if the results violate submission rules.
    Only validates strict rules for full 100-row submissions.
    """
    if len(results) == 0:
        raise ValueError("No results to write.")

    # Only enforce 100-row rule for final submissions
    if len(results) != 100:
        logger.warning(f"Submission has {len(results)} rows (100 required for final submission)")
        return  # Don't raise — allow test runs with fewer candidates

    seen_ids: set[str] = set()
    seen_ranks: set[int] = set()
    prev_score = float("inf")

    for r in results:
        if not CANDIDATE_ID_RE.match(r.candidate_id):
            raise ValueError(f"Invalid candidate_id format: {r.candidate_id}")
        if r.candidate_id in seen_ids:
            raise ValueError(f"Duplicate candidate_id: {r.candidate_id}")
        seen_ids.add(r.candidate_id)

        if r.rank in seen_ranks:
            raise ValueError(f"Duplicate rank: {r.rank}")
        if not (1 <= r.rank <= 100):
            raise ValueError(f"Rank out of range: {r.rank}")
        seen_ranks.add(r.rank)

        if r.score > prev_score + 1e-9:
            raise ValueError(
                f"Score not non-increasing: rank {r.rank} score {r.score} > prev {prev_score}"
            )
        prev_score = r.score

    missing = set(range(1, 101)) - seen_ranks
    if missing:
        raise ValueError(f"Missing ranks: {sorted(missing)}")
