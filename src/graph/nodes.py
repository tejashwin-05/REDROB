"""
Individual LangGraph node functions.
Each node takes RankingState and returns a partial update dict.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.data.loader import CandidateLoader
from src.data.jd_reader import read_job_description
from src.models.candidate import Candidate
from src.models.scoring import CandidateScore
from src.pipeline.jd_parser import parse_job_description
from src.pipeline.embedding_filter import EmbeddingFilter
from src.pipeline.signal_scorer import SignalScorer
from src.pipeline.hybrid_ranker import HybridRanker
from src.pipeline.llm_reranker import LLMReranker
from src.pipeline.output_writer import write_submission
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _time(node_name: str, start: float, state: dict) -> dict:
    elapsed = round(time.time() - start, 2)
    timings = dict(state.get("node_timings") or {})
    timings[node_name] = elapsed
    return {"node_timings": timings}


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — Load JD
# ─────────────────────────────────────────────────────────────────────────────
def load_jd(state: dict) -> dict:
    t = time.time()
    logger.info("═" * 50)
    logger.info("NODE 1 — Load Job Description")
    logger.info("═" * 50)
    try:
        jd = read_job_description(state["jd_path"])
        return {
            "raw_jd": jd,
            "status": "jd_loaded",
            **_time("load_jd", t, state),
        }
    except Exception as e:
        logger.error(f"load_jd failed: {e}")
        return {"errors": (state.get("errors") or []) + [str(e)], "status": "error"}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — Parse JD with LLM
# ─────────────────────────────────────────────────────────────────────────────
def parse_jd(state: dict) -> dict:
    t = time.time()
    logger.info("═" * 50)
    logger.info("NODE 2 — Parse JD with LLM (Groq)")
    logger.info("═" * 50)
    try:
        parsed = parse_job_description(state["raw_jd"], state["groq_api_key"])
        return {
            "parsed_jd": parsed,
            "status": "jd_parsed",
            **_time("parse_jd", t, state),
        }
    except Exception as e:
        logger.error(f"parse_jd failed: {e}")
        return {"errors": (state.get("errors") or []) + [str(e)], "status": "error"}


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — Load Candidates
# ─────────────────────────────────────────────────────────────────────────────
def load_candidates(state: dict) -> dict:
    t = time.time()
    logger.info("═" * 50)
    logger.info("NODE 3 — Load Candidates (100K)")
    logger.info("═" * 50)
    try:
        loader = CandidateLoader(state["candidates_path"])
        candidates = loader.load_all()
        return {
            "all_candidates": candidates,
            "status": "candidates_loaded",
            **_time("load_candidates", t, state),
        }
    except Exception as e:
        logger.error(f"load_candidates failed: {e}")
        return {"errors": (state.get("errors") or []) + [str(e)], "status": "error"}


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — Embedding Filter (FAISS)
# ─────────────────────────────────────────────────────────────────────────────
def embedding_filter(state: dict) -> dict:
    t = time.time()
    logger.info("═" * 50)
    logger.info("NODE 4 — Semantic Embedding Filter (TF-IDF + BGE)")
    logger.info("═" * 50)
    try:
        candidates: List[Candidate] = state["all_candidates"]
        parsed_jd = state["parsed_jd"]
        cache_dir = state.get("cache_dir", "cache")
        top_k = state.get("top_k_embed", 500)
        rebuild = state.get("rebuild_index", False)

        ef = EmbeddingFilter(cache_dir=cache_dir)

        # Try cache first; build if missing or forced rebuild
        loaded = False if rebuild else ef.load_index(candidates)
        if not loaded:
            ef.build_index(candidates)
        else:
            logger.info("Using cached TF-IDF index — skipping re-build")

        semantic_results = ef.search(parsed_jd, top_k=top_k)
        return {
            "semantic_results": semantic_results,
            "status": "embedding_filtered",
            **_time("embedding_filter", t, state),
        }
    except Exception as e:
        logger.error(f"embedding_filter failed: {e}")
        return {"errors": (state.get("errors") or []) + [str(e)], "status": "error"}


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 — Signal Scorer
# ─────────────────────────────────────────────────────────────────────────────
def signal_scorer_node(state: dict) -> dict:
    t = time.time()
    logger.info("═" * 50)
    logger.info("NODE 5 — Signal Scorer (Redrob Behavioral Signals)")
    logger.info("═" * 50)
    try:
        semantic_results: List[Tuple[Candidate, float]] = state["semantic_results"]
        parsed_jd = state["parsed_jd"]

        scorer = SignalScorer(parsed_jd)
        candidates_only = [c for c, _ in semantic_results]
        scored = scorer.score_batch(candidates_only)

        signal_map = {c.candidate_id: score for c, score in scored}
        return {
            "signal_scores": signal_map,
            "status": "signals_scored",
            **_time("signal_scorer", t, state),
        }
    except Exception as e:
        logger.error(f"signal_scorer failed: {e}")
        return {"errors": (state.get("errors") or []) + [str(e)], "status": "error"}


# ─────────────────────────────────────────────────────────────────────────────
# Node 6 — Hybrid Ranker
# ─────────────────────────────────────────────────────────────────────────────
def hybrid_ranker_node(state: dict) -> dict:
    t = time.time()
    logger.info("═" * 50)
    logger.info("NODE 6 — Hybrid Ranker (Semantic + Signals + Skill Match)")
    logger.info("═" * 50)
    try:
        semantic_results: List[Tuple[Candidate, float]] = state["semantic_results"]
        signal_scores: Dict[str, float] = state["signal_scores"]
        parsed_jd = state["parsed_jd"]
        top_n = state.get("top_n_hybrid", 150)

        ranker = HybridRanker(parsed_jd)

        combined = [
            (c, sem_score, signal_scores.get(c.candidate_id, 0.0))
            for c, sem_score in semantic_results
        ]
        hybrid_scores = ranker.rank(combined, top_n=top_n)

        return {
            "hybrid_scores": hybrid_scores,
            "status": "hybrid_ranked",
            **_time("hybrid_ranker", t, state),
        }
    except Exception as e:
        logger.error(f"hybrid_ranker failed: {e}")
        return {"errors": (state.get("errors") or []) + [str(e)], "status": "error"}


# ─────────────────────────────────────────────────────────────────────────────
# Node 7 — LLM Reranker
# ─────────────────────────────────────────────────────────────────────────────
def llm_reranker_node(state: dict) -> dict:
    t = time.time()
    logger.info("═" * 50)
    logger.info("NODE 7 — LLM Reranker (Groq llama-3.3-70b)")
    logger.info("═" * 50)
    try:
        hybrid_scores: List[CandidateScore] = state["hybrid_scores"]
        parsed_jd = state["parsed_jd"]
        groq_api_key = state["groq_api_key"]
        top_n_final = state.get("top_n_final", 100)
        all_candidates: List[Candidate] = state["all_candidates"]

        # Build ID → Candidate map
        id_map = {c.candidate_id: c for c in all_candidates}

        # Get candidates in hybrid rank order
        ranked_candidates = [
            id_map[hs.candidate_id]
            for hs in hybrid_scores
            if hs.candidate_id in id_map
        ]

        reranker = LLMReranker(parsed_jd, groq_api_key)
        final_scores = reranker.rerank(
            candidates=ranked_candidates,
            hybrid_scores=hybrid_scores,
            top_n=top_n_final,
        )

        return {
            "final_scores": final_scores,
            "status": "llm_reranked",
            **_time("llm_reranker", t, state),
        }
    except Exception as e:
        logger.error(f"llm_reranker failed: {e}")
        return {"errors": (state.get("errors") or []) + [str(e)], "status": "error"}


# ─────────────────────────────────────────────────────────────────────────────
# Node 8 — Write Submission
# ─────────────────────────────────────────────────────────────────────────────
def write_output(state: dict) -> dict:
    t = time.time()
    logger.info("═" * 50)
    logger.info("NODE 8 — Write Submission CSV")
    logger.info("═" * 50)
    try:
        final_scores: List[CandidateScore] = state["final_scores"]
        output_path = state["output_path"]

        path = write_submission(final_scores, output_path)

        logger.info(f"Submission ready → {path}")
        return {
            "submission_path": str(path),
            "status": "complete",
            **_time("write_output", t, state),
        }
    except Exception as e:
        logger.error(f"write_output failed: {e}")
        return {"errors": (state.get("errors") or []) + [str(e)], "status": "error"}
