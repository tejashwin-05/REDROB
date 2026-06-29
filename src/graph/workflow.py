"""
LangGraph Workflow — wires all 8 nodes into a directed acyclic graph.

The graph is also LangGraph Studio compatible:
  - Each node is a named step visible in the studio
  - State is fully typed (TypedDict)
  - The graph is compiled and exported as `graph` at module level
    (the name LangGraph Studio looks for)

Graph topology (linear pipeline):
  load_jd → parse_jd → load_candidates → embedding_filter
          → signal_scorer → hybrid_ranker → llm_reranker → write_output
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from src.graph.state import RankingState
from src.graph.nodes import (
    load_jd,
    parse_jd,
    load_candidates,
    embedding_filter,
    signal_scorer_node,
    hybrid_ranker_node,
    llm_reranker_node,
    write_output,
)


def build_ranking_graph() -> StateGraph:
    """
    Build and compile the full ranking pipeline as a LangGraph StateGraph.
    Returns a compiled graph ready for invocation.
    """
    builder = StateGraph(RankingState)

    # ── Register nodes ─────────────────────────────────────────────────────
    builder.add_node("load_jd", load_jd)
    builder.add_node("parse_jd", parse_jd)
    builder.add_node("load_candidates", load_candidates)
    builder.add_node("embedding_filter", embedding_filter)
    builder.add_node("signal_scorer", signal_scorer_node)
    builder.add_node("hybrid_ranker", hybrid_ranker_node)
    builder.add_node("llm_reranker", llm_reranker_node)
    builder.add_node("write_output", write_output)

    # ── Entry point ─────────────────────────────────────────────────────────
    builder.set_entry_point("load_jd")

    # ── Edges (sequential pipeline) ─────────────────────────────────────────
    builder.add_edge("load_jd", "parse_jd")
    builder.add_edge("parse_jd", "load_candidates")
    builder.add_edge("load_candidates", "embedding_filter")
    builder.add_edge("embedding_filter", "signal_scorer")
    builder.add_edge("signal_scorer", "hybrid_ranker")
    builder.add_edge("hybrid_ranker", "llm_reranker")
    builder.add_edge("llm_reranker", "write_output")
    builder.add_edge("write_output", END)

    return builder.compile()


# ─── LangGraph Studio entry point ───────────────────────────────────────────
# LangGraph Studio looks for a compiled graph named `graph` in the module
# that `langgraph.json` points to.
graph = build_ranking_graph()
