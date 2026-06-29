"""Efficient streaming loader for the 100K candidates JSONL file.
Also handles JSON array files (e.g. sample_candidates.json).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator, Iterator, List, Optional

from src.models.candidate import Candidate
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CandidateLoader:
    """
    Lazy loader for the candidates dataset.
    Supports:
      - .jsonl: one JSON object per line (streaming, memory-efficient)
      - .json:  a top-level JSON array (loaded fully)
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Candidates file not found: {self.path}")
        self._is_jsonl = self.path.suffix.lower() == ".jsonl"
        logger.info(f"CandidateLoader initialised → {self.path} ({self.path.stat().st_size / 1e6:.1f} MB)")

    def stream(self) -> Generator[Candidate, None, None]:
        """Stream candidates one at a time — memory efficient for JSONL."""
        if self._is_jsonl:
            yield from self._stream_jsonl()
        else:
            yield from self._stream_json_array()

    def _stream_jsonl(self) -> Generator[Candidate, None, None]:
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield Candidate.model_validate(json.loads(line))
                except Exception as e:
                    logger.warning(f"Skipping malformed record: {e}")

    def _stream_json_array(self) -> Generator[Candidate, None, None]:
        with open(self.path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for item in data:
            try:
                yield Candidate.model_validate(item)
            except Exception as e:
                logger.warning(f"Skipping malformed record: {e}")

    def load_all(self, limit: Optional[int] = None) -> List[Candidate]:
        """Load all candidates into memory."""
        logger.info(f"Loading candidates (limit={limit or 'all'})...")
        candidates = []
        for i, c in enumerate(self.stream()):
            if limit and i >= limit:
                break
            candidates.append(c)
        logger.info(f"Loaded {len(candidates)} candidates.")
        return candidates

    def load_ids_and_texts(
        self,
        text_builder,
        limit: Optional[int] = None,
    ) -> tuple[List[str], List[str], List[Candidate]]:
        """
        Stream the dataset and return (ids, texts, candidates).
        text_builder: callable(Candidate) -> str
        """
        ids, texts, candidates = [], [], []
        for i, candidate in enumerate(self.stream()):
            if limit and i >= limit:
                break
            ids.append(candidate.candidate_id)
            texts.append(text_builder(candidate))
            candidates.append(candidate)
            if (i + 1) % 10000 == 0:
                logger.info(f"  Streamed {i + 1} candidates...")
        logger.info(f"Stream complete — {len(ids)} candidates.")
        return ids, texts, candidates
