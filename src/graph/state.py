"""
LangGraph State definition for the REDROB ranking workflow.
All nodes communicate through this typed state object.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, Tuple
from typing_extensions import TypedDict

from src.models.candidate import Candidate
from src.models.job_description import JobDescription, ParsedJD
from src.models.scoring import CandidateScore


class RankingState(TypedDict):
    """
    Shared state passed between all nodes in the LangGraph workflow.
    Each field is written by one node and read by subsequent nodes.
    """

    # ── Input ──────────────────────────────────────────────────────────────
    jd_path: str                          # Path to the job description file
    candidates_path: str                  # Path to candidates.jsonl
    output_path: str                      # Where to write submission.csv
    groq_api_key: str                     # Groq API key
    cache_dir: str                        # Directory for FAISS cache
    top_k_embed: int                      # How many to retrieve via embeddings
    top_n_hybrid: int                     # How many to pass to LLM reranker
    top_n_final: int                      # Final submission count (always 100)
    rebuild_index: bool                   # Force rebuild FAISS index

    # ── Node 1: JD Loader output ────────────────────────────────────────────
    raw_jd: Optional[JobDescription]

    # ── Node 2: JD Parser output ────────────────────────────────────────────
    parsed_jd: Optional[ParsedJD]

    # ── Node 3: Candidate Loader output ─────────────────────────────────────
    all_candidates: Optional[List[Candidate]]

    # ── Node 4: Embedding Filter output ─────────────────────────────────────
    # List of (Candidate, semantic_score)
    semantic_results: Optional[List[Tuple[Candidate, float]]]

    # ── Node 5: Signal Scorer output ─────────────────────────────────────────
    # Dict: candidate_id → signal_score
    signal_scores: Optional[Dict[str, float]]

    # ── Node 6: Hybrid Ranker output ─────────────────────────────────────────
    hybrid_scores: Optional[List[CandidateScore]]

    # ── Node 7: LLM Reranker output ──────────────────────────────────────────
    final_scores: Optional[List[CandidateScore]]

    # ── Node 8: Output Writer output ─────────────────────────────────────────
    submission_path: Optional[str]

    # ── Metadata ──────────────────────────────────────────────────────────────
    errors: List[str]
    status: str                          # e.g. "running", "complete", "error"
    node_timings: Dict[str, float]       # node_name → elapsed_seconds
