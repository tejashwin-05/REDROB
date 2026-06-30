"""
Embedding Filter — semantic retrieval with automatic backend selection:

  Priority 1 (best):  ChromaDB + BAAI/bge-small-en-v1.5 embeddings
                       → true semantic search, cosine similarity
                       → requires cache/chroma_db/ to exist (run embed_candidates.py first)

  Priority 2 (fallback): TF-IDF + cosine similarity
                       → keyword-based, no GPU/model needed
                       → always available, used when ChromaDB not ready

The pipeline prints which backend is active so it's always clear.
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.models.candidate import Candidate
from src.models.job_description import ParsedJD
from src.utils.logging_utils import get_logger
from src.utils.text_utils import build_candidate_text

logger = get_logger(__name__)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ── ChromaDB config ────────────────────────────────────────────────────────
COLLECTION_NAME  = "candidates"
BGE_MODEL_NAME   = "BAAI/bge-small-en-v1.5"
# Query-side prefix for BGE retrieval models
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ── TF-IDF config (fallback) ───────────────────────────────────────────────
TFIDF_MAX_FEATURES = 25000
TFIDF_NGRAM_RANGE  = (1, 2)


class EmbeddingFilter:
    """
    Semantic retrieval over 100K candidates.
    Automatically uses ChromaDB (semantic) if available, otherwise TF-IDF.
    """

    def __init__(self, cache_dir: str | Path = "cache"):
        self.cache_dir  = Path(cache_dir)
        self.chroma_dir = self.cache_dir / "chroma_db"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.candidates: List[Candidate] = []
        self._backend: str = "none"   # "chroma" | "tfidf"

        # TF-IDF state
        self.vectorizer  = None
        self.tfidf_matrix = None
        self.candidate_ids: List[str] = []

        # ChromaDB state
        self._chroma_collection = None
        self._bge_model         = None

    # ── Public API ─────────────────────────────────────────────────────────

    def build_index(self, candidates: List[Candidate]) -> None:
        """Build whichever index is appropriate (called when rebuild_index=True)."""
        self.candidates = candidates
        if self._chroma_ready():
            logger.info("ChromaDB collection exists — no rebuild needed for semantic index.")
            self._init_chroma()
        else:
            logger.info("ChromaDB not found — building TF-IDF index.")
            self._build_tfidf(candidates)

    def load_index(self, candidates: List[Candidate]) -> bool:
        """Load cached index. Returns True if any cache exists."""
        self.candidates = candidates
        if self._chroma_ready():
            self._init_chroma()
            return True
        if self._tfidf_ready():
            self._load_tfidf(candidates)
            return True
        return False

    def search(
        self,
        parsed_jd: ParsedJD,
        top_k: int = 500,
    ) -> List[Tuple[Candidate, float]]:
        """Return top_k (Candidate, score) sorted by similarity descending."""
        if self._backend == "chroma":
            return self._chroma_search(parsed_jd, top_k)
        elif self._backend == "tfidf":
            return self._tfidf_search_results(parsed_jd, top_k)
        else:
            raise RuntimeError("No index loaded. Call build_index() or load_index() first.")

    # ── ChromaDB backend ───────────────────────────────────────────────────

    def _chroma_ready(self) -> bool:
        """Check if ChromaDB collection directory exists and has data."""
        return (self.chroma_dir / "chroma.sqlite3").exists()

    def _init_chroma(self) -> None:
        import chromadb
        logger.info(f"Loading ChromaDB from {self.chroma_dir} ...")
        client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self._chroma_collection = client.get_collection(COLLECTION_NAME)
        count = self._chroma_collection.count()
        logger.info(f"ChromaDB ready — {count} embeddings (semantic backend active)")
        self._backend = "chroma"

    def _load_bge_model(self):
        if self._bge_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading BGE model: {BGE_MODEL_NAME}")
            self._bge_model = SentenceTransformer(BGE_MODEL_NAME, device="cpu")

    def _chroma_search(
        self, parsed_jd: ParsedJD, top_k: int
    ) -> List[Tuple[Candidate, float]]:
        self._load_bge_model()

        query_text = BGE_QUERY_PREFIX + parsed_jd.embedding_text
        query_emb  = self._bge_model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        logger.info(f"ChromaDB semantic search (top-{top_k})...")
        t = time.time()
        results = self._chroma_collection.query(
            query_embeddings=query_emb.tolist(),
            n_results=top_k,
            include=["distances", "metadatas"],
        )
        logger.info(f"  ChromaDB query done in {time.time()-t:.2f}s")

        id_map = {c.candidate_id: c for c in self.candidates}
        output = []
        for cid, dist in zip(results["ids"][0], results["distances"][0]):
            candidate = id_map.get(cid)
            if candidate:
                # ChromaDB cosine distance: 0=identical, 2=opposite
                # Convert to similarity: 1 - distance (range ~0 to 1)
                similarity = float(1.0 - dist)
                output.append((candidate, similarity))

        logger.info(
            f"Semantic search complete — top-{len(output)} candidates "
            f"(ChromaDB / BGE-small)"
        )
        return output

    # ── TF-IDF backend ────────────────────────────────────────────────────

    def _tfidf_path(self)   -> Path: return self.cache_dir / "tfidf_vectorizer.pkl"
    def _matrix_path(self)  -> Path: return self.cache_dir / "tfidf_matrix.npz"
    def _ids_path(self)     -> Path: return self.cache_dir / "candidate_ids.npy"

    def _tfidf_ready(self) -> bool:
        return (
            self._tfidf_path().exists()
            and self._matrix_path().exists()
            and self._ids_path().exists()
        )

    def _build_tfidf(self, candidates: List[Candidate]) -> None:
        import scipy.sparse as sp
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.candidate_ids = [c.candidate_id for c in candidates]
        logger.info(f"Building TF-IDF index for {len(candidates)} candidates...")
        t = time.time()
        texts = [build_candidate_text(c) for c in candidates]
        logger.info(f"  Texts built in {time.time()-t:.1f}s")

        t = time.time()
        self.vectorizer   = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            sublinear_tf=True, min_df=2,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        logger.info(f"  TF-IDF fit in {time.time()-t:.1f}s — shape: {self.tfidf_matrix.shape}")

        with open(self._tfidf_path(), "wb") as f:
            pickle.dump(self.vectorizer, f)
        sp.save_npz(str(self._matrix_path()), self.tfidf_matrix)
        np.save(str(self._ids_path()), np.array(self.candidate_ids))
        logger.info(f"TF-IDF index saved → {self.cache_dir}")
        self._backend = "tfidf"

    def _load_tfidf(self, candidates: List[Candidate]) -> None:
        import scipy.sparse as sp

        logger.info("Loading TF-IDF index from cache...")
        with open(self._tfidf_path(), "rb") as f:
            self.vectorizer = pickle.load(f)
        self.tfidf_matrix  = sp.load_npz(str(self._matrix_path()))
        self.candidate_ids = list(np.load(str(self._ids_path()), allow_pickle=True))
        logger.info(
            f"TF-IDF index loaded — "
            f"{self.tfidf_matrix.shape[0]} candidates, "
            f"{self.tfidf_matrix.shape[1]} features"
        )
        self._backend = "tfidf"

    def _tfidf_search_results(
        self, parsed_jd: ParsedJD, top_k: int
    ) -> List[Tuple[Candidate, float]]:
        from sklearn.metrics.pairwise import cosine_similarity

        id_map = {c.candidate_id: c for c in self.candidates}
        q      = self.vectorizer.transform([parsed_jd.embedding_text])

        logger.info(f"TF-IDF search (top-{top_k})...")
        t      = time.time()
        scores = cosine_similarity(q, self.tfidf_matrix)[0]
        top_ix = scores.argsort()[::-1][:top_k]
        logger.info(f"  TF-IDF search done in {time.time()-t:.2f}s")

        results = [
            (id_map[self.candidate_ids[i]], float(scores[i]))
            for i in top_ix
            if self.candidate_ids[i] in id_map
        ]
        logger.info(
            f"Keyword search complete — top-{len(results)} candidates (TF-IDF fallback)"
        )
        return results
