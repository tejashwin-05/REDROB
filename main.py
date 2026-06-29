"""
REDROB — AI Candidate Ranking System
Entry point for running the full pipeline.

Usage:
    uv run python main.py [--rebuild-index] [--top-k 500] [--top-n 150]

Or to invoke directly via the graph:
    uv run python main.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env ─────────────────────────────────────────────────────────────
load_dotenv()

# ── Ensure src is on path ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.graph.workflow import build_ranking_graph
from src.utils.logging_utils import get_logger

logger = get_logger("main")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "India_runs_data_and_ai_challenge"
CACHE_DIR = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REDROB AI Candidate Ranking System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--candidates",
        default=str(DATA_DIR / "candidates.jsonl"),
        help="Path to candidates.jsonl",
    )
    parser.add_argument(
        "--jd",
        default=str(DATA_DIR / "job_description.docx"),
        help="Path to job description file (.docx or .txt)",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "submission.csv"),
        help="Path for submission CSV output",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(CACHE_DIR),
        help="Directory for FAISS index cache",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=500,
        help="Number of candidates to retrieve via embedding search",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=150,
        help="Number of candidates to pass to LLM reranker",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        default=False,
        help="Force rebuild of FAISS index (ignore cache)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("GROQ_API_KEY not set. Please add it to .env or export it.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  REDROB — AI Candidate Ranking System")
    logger.info("=" * 60)
    logger.info(f"  Candidates : {args.candidates}")
    logger.info(f"  JD         : {args.jd}")
    logger.info(f"  Output     : {args.output}")
    logger.info(f"  Cache      : {args.cache_dir}")
    logger.info(f"  Top-K embed: {args.top_k}")
    logger.info(f"  Top-N LLM  : {args.top_n}")
    logger.info(f"  Rebuild idx: {args.rebuild_index}")
    logger.info("=" * 60)

    # Build the LangGraph pipeline
    graph = build_ranking_graph()

    # Initial state
    initial_state = {
        "jd_path": args.jd,
        "candidates_path": args.candidates,
        "output_path": args.output,
        "groq_api_key": groq_api_key,
        "cache_dir": args.cache_dir,
        "top_k_embed": args.top_k,
        "top_n_hybrid": args.top_n,
        "top_n_final": 100,
        "rebuild_index": args.rebuild_index,
        "raw_jd": None,
        "parsed_jd": None,
        "all_candidates": None,
        "semantic_results": None,
        "signal_scores": None,
        "hybrid_scores": None,
        "final_scores": None,
        "submission_path": None,
        "errors": [],
        "status": "starting",
        "node_timings": {},
    }

    # Run the graph
    wall_start = time.time()
    final_state = graph.invoke(initial_state)
    wall_elapsed = round(time.time() - wall_start, 1)

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Status   : {final_state.get('status', 'unknown')}")
    logger.info(f"  Output   : {final_state.get('submission_path', 'N/A')}")
    logger.info(f"  Total    : {wall_elapsed}s")
    logger.info("")
    logger.info("  Node timings:")
    for node, elapsed in (final_state.get("node_timings") or {}).items():
        logger.info(f"    {node:<25} {elapsed:>7.1f}s")

    errors = final_state.get("errors") or []
    if errors:
        logger.error("\n  Errors encountered:")
        for e in errors:
            logger.error(f"    - {e}")
        sys.exit(1)

    logger.info("\n  Done. Validate with:")
    logger.info(f"    uv run python India_runs_data_and_ai_challenge/validate_submission.py {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
