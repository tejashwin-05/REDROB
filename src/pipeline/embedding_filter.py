"""
Embedding Filter — two-stage semantic retrieval:

Stage 1 (fast): TF-IDF + cosine similarity over all 100K candidates (~2 min)
  → Retrieves top-1000 candidates

Stage 2 (quality): fastembed ONNX (BAAI/bge-small-en-v1.5) over top-1000 only
  → Reranks to produce final top-K (default 500)

Stage 2 is skipped and Stage 1 scores are used if fastembed is too slow.

The TF-IDF index is cached (pickle) for fast re-use on repeated runs.
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.models.candidate import Candidate
from src.models.job_description import ParsedJD
from src.utils.logging_utils import get_logger
from src.utils.text_utils import build_candidate_text

logger = get_logger(__name__)

# Suppress HuggingFace symlink warning on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# TF-IDF config
TFIDF_MAX_FEATURES = 25000
TFIDF_NGRAM_RANGE = (1, 2)


class EmbeddingFilter:
    """
    Two-stage semantic filter:
    1. TF-IDF over 100K (fast, cached)
    2. fastembed ONNX rerank of top-1K (quality, runs on small set)
    """

    def __init__(
        self,
        cache_dir: str | Path = "cache",
        stage2_top: int = 1000,   # how many to rerank in stage 2
        enable_stage2: bool = False,  # disabled by default — LLM is the quality gate
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stage2_top = stage2_top
        self.enable_stage2 = enable_stage2

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None          # sparse (n_candidates, n_features)
        self.candidate_ids: List[str] = []
        self.candidates: List[Candidate] = []

    # ── Cache paths ────────────────────────────────────────────────────────
    def _tfidf_path(self) -> Path:
        return self.cache_dir / "tfidf_vectorizer.pkl"

    def _matrix_path(self) -> Path:
        return self.cache_dir / "tfidf_matrix.npz"

    def _ids_path(self) -> Path:
        return self.cache_dir / "candidate_ids.npy"

    # ── Stage 1: TF-IDF ────────────────────────────────────────────────────
    def build_index(self, candidates: List[Candidate]) -> None:
        """Build TF-IDF index over all candidates and save to cache."""
        import scipy.sparse as sp

        self.candidates = candidates
        self.candidate_ids = [c.candidate_id for c in candidates]

        logger.info(f"Building TF-IDF index for {len(candidates)} candidates...")
        t = time.time()
        texts = [build_candidate_text(c) for c in candidates]
        logger.info(f"  Texts built in {time.time()-t:.1f}s")

        t = time.time()
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            sublinear_tf=True,
            min_df=2,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        logger.info(
            f"  TF-IDF fit in {time.time()-t:.1f}s — "
            f"shape: {self.tfidf_matrix.shape}"
        )

        # Save
        with open(self._tfidf_path(), "wb") as f:
            pickle.dump(self.vectorizer, f)
        sp.save_npz(str(self._matrix_path()), self.tfidf_matrix)
        np.save(str(self._ids_path()), np.array(self.candidate_ids))
        logger.info(f"TF-IDF index saved → {self.cache_dir}")

    def load_index(self, candidates: List[Candidate]) -> bool:
        """Load cached TF-IDF index. Returns True if cache hit."""
        import scipy.sparse as sp

        if not (
            self._tfidf_path().exists()
            and self._matrix_path().exists()
            and self._ids_path().exists()
        ):
            return False

        logger.info("Loading TF-IDF index from cache...")
        with open(self._tfidf_path(), "rb") as f:
            self.vectorizer = pickle.load(f)
        self.tfidf_matrix = sp.load_npz(str(self._matrix_path()))
        self.candidate_ids = list(np.load(str(self._ids_path()), allow_pickle=True))
        self.candidates = candidates
        logger.info(
            f"TF-IDF index loaded — "
            f"{self.tfidf_matrix.shape[0]} candidates, "
            f"{self.tfidf_matrix.shape[1]} features"
        )
        return True

    def _tfidf_search(self, jd_text: str, top_k: int) -> List[Tuple[str, float]]:
        """Return top_k (candidate_id, score) by TF-IDF cosine similarity."""
        assert self.vectorizer is not None, "Index not built."
        q = self.vectorizer.transform([jd_text])
        scores = cosine_similarity(q, self.tfidf_matrix)[0]
        top_indices = scores.argsort()[::-1][:top_k]
        return [(self.candidate_ids[i], float(scores[i])) for i in top_indices]

    # ── Stage 2: fastembed rerank ──────────────────────────────────────────
    def _fastembed_rerank(
        self,
        candidates_subset: List[Candidate],
        jd: ParsedJD,
        top_k: int,
    ) -> List[Tuple[Candidate, float]]:
        """Rerank a small set using fastembed ONNX embeddings."""
        try:
            from fastembed import TextEmbedding

            logger.info(
                f"Stage 2: fastembed reranking {len(candidates_subset)} candidates..."
            )
            model = TextEmbedding(FASTEMBED_MODEL)

            # Encode candidates
            texts = [build_candidate_text(c) for c in candidates_subset]
            t = time.time()
            cand_embs = np.array(list(model.embed(texts)), dtype="float32")
            logger.info(f"  Candidates encoded in {time.time()-t:.1f}s")

            # Encode query
            query_text = BGE_QUERY_PREFIX + jd.embedding_text
            query_emb = np.array(list(model.embed([query_text])), dtype="float32")

            # Cosine similarity (already L2-normalised by fastembed)
            sims = (cand_embs @ query_emb.T).flatten()
            top_indices = sims.argsort()[::-1][:top_k]

            result = [
                (candidates_subset[i], float(sims[i]))
                for i in top_indices
            ]
            logger.info(f"  Stage 2 complete — top-{len(result)} returned")
            return result

        except Exception as e:
            logger.warning(f"Stage 2 (fastembed) failed: {e} — using stage 1 scores only")
            return []

    # ── Public API ─────────────────────────────────────────────────────────
    def search(
        self,
        parsed_jd: ParsedJD,
        top_k: int = 500,
    ) -> List[Tuple[Candidate, float]]:
        """
        Run two-stage search and return (Candidate, score) sorted descending.
        """
        assert self.vectorizer is not None, "Index not built or loaded."

        id_map = {c.candidate_id: c for c in self.candidates}

        # Stage 1: TF-IDF → top-1000
        stage1_top = max(top_k, self.stage2_top)
        logger.info(f"Stage 1: TF-IDF search (top-{stage1_top})...")
        t = time.time()
        stage1_results = self._tfidf_search(parsed_jd.embedding_text, stage1_top)
        logger.info(f"  Stage 1 done in {time.time()-t:.2f}s")

        # Build candidate subset for stage 2
        stage1_candidates = [
            id_map[cid]
            for cid, _ in stage1_results
            if cid in id_map
        ]
        stage1_score_map = {cid: s for cid, s in stage1_results}

        if self.enable_stage2 and len(stage1_candidates) > 0:
            # Stage 2: fastembed rerank top-1000
            stage2_results = self._fastembed_rerank(
                stage1_candidates[:self.stage2_top],
                parsed_jd,
                top_k=top_k,
            )
            if stage2_results:
                logger.info(
                    f"Embedding search complete — "
                    f"top-{len(stage2_results)} from stage 2 (fastembed)"
                )
                return stage2_results

        # Fall back to stage 1 scores only
        results = [
            (id_map[cid], score)
            for cid, score in stage1_results[:top_k]
            if cid in id_map
        ]
        logger.info(
            f"Embedding search complete — "
            f"top-{len(results)} from stage 1 (TF-IDF)"
        )
        return results
