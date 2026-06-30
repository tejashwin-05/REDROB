"""
Embed candidates.jsonl into ChromaDB using BAAI/bge-small-en-v1.5.

Stores 100K candidate embeddings (384-dim) in a persistent ChromaDB
collection so the ranking pipeline can do true semantic search instead
of TF-IDF keyword matching.

Usage:
    uv run python embed_candidates.py

Output:
    cache/chroma_db/   — persistent ChromaDB collection (~600MB)

Runtime estimate:
    - CPU only, single-process: ~20-40 min (depends on cores / clock speed)
    - Multi-process pool: uses all CPU cores, ~10-20 min
    - Progress is printed every 5K candidates so you can monitor
"""

import json
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
INPUT_PATH = BASE_DIR / "India_runs_data_and_ai_challenge" / "candidates.jsonl"
CHROMA_DIR = BASE_DIR / "cache" / "chroma_db"

# ── Config ─────────────────────────────────────────────────────────────────
COLLECTION_NAME  = "candidates"
MODEL_NAME       = "BAAI/bge-small-en-v1.5"
ENCODE_BATCH_SIZE = 32   # Small batches are faster on CPU (avoids memory thrashing)
CHROMA_BATCH_SIZE = 1000

# BGE passage prefix (query uses a different prefix at search time)
PASSAGE_PREFIX = "Represent this document for retrieval: "


def flatten_record(rec: dict) -> tuple[str, dict]:
    """
    Build one rich searchable text blob + flat metadata dict per candidate.
    Metadata fields are kept flat (no nested dicts) — ChromaDB requirement.
    """
    profile  = rec.get("profile", {}) or {}
    career   = rec.get("career_history", []) or []
    education = rec.get("education", []) or []
    skills   = rec.get("skills", []) or []
    certs    = rec.get("certifications", []) or []
    languages = rec.get("languages", []) or []
    signals  = rec.get("redrob_signals", {}) or {}

    parts = []

    if profile.get("headline"):
        parts.append(f"Headline: {profile['headline']}")
    if profile.get("summary"):
        parts.append(f"Summary: {profile['summary']}")
    if profile.get("current_title") or profile.get("current_company"):
        parts.append(
            f"Current role: {profile.get('current_title', '')} at "
            f"{profile.get('current_company', '')} "
            f"({profile.get('current_industry', '')})"
        )
    if profile.get("years_of_experience") is not None:
        parts.append(f"Years of experience: {profile['years_of_experience']}")
    if profile.get("location") or profile.get("country"):
        parts.append(f"Location: {profile.get('location', '')}, {profile.get('country', '')}")

    if career:
        career_lines = []
        for job in career:
            line = (
                f"{job.get('title', '')} at {job.get('company', '')} "
                f"({job.get('industry', '')}, {job.get('duration_months', '?')} months): "
                f"{job.get('description', '')[:300]}"
            )
            career_lines.append(line)
        parts.append("Career history: " + " | ".join(career_lines))

    if education:
        edu_lines = [
            f"{e.get('degree', '')} in {e.get('field_of_study', '')} "
            f"from {e.get('institution', '')} ({e.get('tier', '')})"
            for e in education
        ]
        parts.append("Education: " + " | ".join(edu_lines))

    if skills:
        skill_names = [s.get("name", "") for s in skills if s.get("name")]
        parts.append("Skills: " + ", ".join(skill_names))

    if certs:
        cert_names = [
            c.get("name", str(c)) if isinstance(c, dict) else str(c)
            for c in certs
        ]
        parts.append("Certifications: " + ", ".join(cert_names))

    if languages:
        lang_names = [l.get("language", "") for l in languages if l.get("language")]
        parts.append("Languages: " + ", ".join(lang_names))

    text = "\n".join(parts)

    # Flat metadata for ChromaDB filtering
    metadata = {
        "candidate_id":       rec.get("candidate_id", ""),
        "name":               profile.get("anonymized_name", ""),
        "current_title":      profile.get("current_title", ""),
        "current_company":    profile.get("current_company", ""),
        "location":           profile.get("location", ""),
        "country":            profile.get("country", ""),
        "years_of_experience": float(profile.get("years_of_experience", 0.0)),
        "open_to_work":       bool(signals.get("open_to_work_flag", False)),
        "notice_period_days": int(signals.get("notice_period_days", 0)),
        "preferred_work_mode": signals.get("preferred_work_mode", ""),
    }
    return text, metadata


def load_records(path: Path) -> tuple[list, list, list]:
    ids, texts, metadatas = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text, meta = flatten_record(rec)
            ids.append(rec.get("candidate_id"))
            texts.append(text)
            metadatas.append(meta)
            if (i + 1) % 10000 == 0:
                print(f"  Loaded {i + 1} records...")
    return ids, texts, metadatas


def main():
    t0 = time.time()

    # ── Check if collection already exists and is complete ─────────────────
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    existing = None
    try:
        existing = client.get_collection(COLLECTION_NAME)
        count = existing.count()
        if count >= 100000:
            print(f"Collection '{COLLECTION_NAME}' already has {count} embeddings — nothing to do.")
            print("You can now run: uv run python main.py")
            return
        else:
            print(f"Existing collection has {count} embeddings — rebuilding from scratch.")
            client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # Collection doesn't exist yet

    # ── Load records ────────────────────────────────────────────────────────
    print(f"Loading and flattening records from {INPUT_PATH} ...")
    ids, texts, metadatas = load_records(INPUT_PATH)
    print(f"Loaded {len(ids)} records in {time.time() - t0:.1f}s")

    # ── Encode ─────────────────────────────────────────────────────────────
    print(f"\nLoading model {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    prefixed_texts = [PASSAGE_PREFIX + t for t in texts]
    print(f"Encoding {len(prefixed_texts)} texts (this takes ~20-40 min on CPU)...")
    print("Using multi-process pool across all CPU cores...")

    t_enc = time.time()
    print("Using single-process encode with progress bar...")
    embeddings = model.encode(
        prefixed_texts,
        batch_size=ENCODE_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print(f"Encoded {len(embeddings)} vectors in {time.time() - t_enc:.1f}s")
    print(f"Embedding shape: {embeddings.shape}")

    # ── Write to ChromaDB ───────────────────────────────────────────────────
    print(f"\nWriting to ChromaDB at {CHROMA_DIR} ...")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )

    for i in range(0, len(ids), CHROMA_BATCH_SIZE):
        batch_ids   = ids[i : i + CHROMA_BATCH_SIZE]
        batch_embs  = embeddings[i : i + CHROMA_BATCH_SIZE].tolist()
        batch_docs  = texts[i : i + CHROMA_BATCH_SIZE]
        batch_meta  = metadatas[i : i + CHROMA_BATCH_SIZE]
        collection.add(
            ids=batch_ids,
            embeddings=batch_embs,
            documents=batch_docs,
            metadatas=batch_meta,
        )
        print(f"  Inserted {min(i + CHROMA_BATCH_SIZE, len(ids))}/{len(ids)}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Collection count: {collection.count()}")
    print(f"ChromaDB saved at: {CHROMA_DIR}")
    print("\nNext step — run the ranking pipeline:")
    print("  uv run python main.py")


if __name__ == "__main__":
    main()
