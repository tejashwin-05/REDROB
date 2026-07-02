# REDROB — AI Candidate Ranking System

An intelligent candidate ranking pipeline that ranks 100K candidates the way a great recruiter would — using semantic understanding, behavioral signals, and LLM reasoning.

---

## Architecture

```
Job Description (.docx)
        │
        ▼
┌─────────────────┐
│  1. load_jd     │  Read job description file (.docx / .txt)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. parse_jd    │  Groq LLM extracts structured requirements:
└────────┬────────┘  skills, experience, responsibilities, disqualifiers
         │
         ▼
┌──────────────────────┐
│  3. load_candidates  │  Stream 100K JSONL candidates into memory (~9s)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  4. embedding_filter │  ChromaDB + BAAI/bge-small-en-v1.5:
└──────────┬───────────┘  semantic search over 100K in 0.06s → top-500
           │              (falls back to TF-IDF if ChromaDB not built)
           ▼
┌──────────────────────┐
│  5. signal_scorer    │  Score 6 Redrob behavioral signal components:
└──────────┬───────────┘  engagement, reliability, availability,
           │              profile quality, activity, skill validation
           ▼
┌──────────────────────┐
│  6. hybrid_ranker    │  Fuse scores:
└──────────┬───────────┘  45% semantic + 30% signals + 25% skill match
           │              + experience gate → top-150
           ▼
┌──────────────────────┐
│  7. llm_reranker     │  Groq llama-3.3-70b reranks top-150 in batches
└──────────┬───────────┘  Final = 60% LLM score + 40% hybrid score
           │
           ▼
┌──────────────────────┐
│  8. write_output     │  Validates + writes submission.csv (100 rows)
└──────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph (StateGraph) |
| LLM | Groq `llama-3.3-70b-versatile` |
| Embeddings | `BAAI/bge-small-en-v1.5` (sentence-transformers, free, local CPU) |
| Vector DB | ChromaDB (persistent, HNSW index) |
| Fallback Retrieval | TF-IDF + cosine similarity (scikit-learn) |
| Fuzzy Skill Match | RapidFuzz |
| Data Models | Pydantic v2 |
| UI | Streamlit |
| Visualisation | Plotly |
| Package Manager | uv |
| Python | 3.14 |

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/tejashwin-05/REDROB.git
cd REDROB
uv sync

# 2. Add your Groq API key
cp .env.example .env
# Edit .env → GROQ_API_KEY="your_key_here"

# 3. Build the semantic index (one-time, ~5 hours on CPU)
#    Embeds all 100K candidates into ChromaDB using BGE-small
uv run python embed_candidates.py

# 4. Launch the UI (recommended)
uv run streamlit run app.py

# 5. Or run the pipeline directly from CLI
uv run python main.py

# 6. Validate the output
uv run python India_runs_data_and_ai_challenge/validate_submission.py output/submission.csv
```

---

## Commands

```bash
# Launch the recruiter UI (recommended entry point)
uv run streamlit run app.py
# Opens at http://localhost:8501
# Features: PDF JD upload, live pipeline progress, rankings, analytics, candidate deep dive

# Run pipeline directly from CLI
uv run python main.py

# First run — build TF-IDF fallback index (if ChromaDB not ready)
uv run python main.py --rebuild-index

# Normal run — uses ChromaDB if available, TF-IDF otherwise
uv run python main.py

# Custom options
uv run python main.py --top-k 500 --top-n 150 --output my_submission.csv

# LangGraph Studio — visualise the workflow graph
uv run langgraph dev
```

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--candidates` | `India_runs_data_and_ai_challenge/candidates.jsonl` | Path to candidates dataset |
| `--jd` | `India_runs_data_and_ai_challenge/job_description.docx` | Job description file |
| `--output` | `output/submission.csv` | Output CSV path |
| `--cache-dir` | `cache/` | Directory for ChromaDB and TF-IDF cache |
| `--top-k` | `500` | Candidates retrieved by embedding search |
| `--top-n` | `150` | Candidates passed to LLM reranker |
| `--rebuild-index` | `False` | Force rebuild TF-IDF index |

---

## Two-Step Setup: Embeddings + Pipeline

### Step 1 — Build ChromaDB (one-time)

```bash
uv run python embed_candidates.py
```

- Reads all 100K candidates from `candidates.jsonl`
- Encodes each candidate into a 384-dim vector using `BAAI/bge-small-en-v1.5`
- Stores vectors + metadata in `cache/chroma_db/` (~600MB)
- Takes ~5 hours on CPU (once done, never needs to run again)
- If `cache/chroma_db/` already exists with 100K entries, it skips automatically

### Step 2 — Run the Pipeline

```bash
uv run python main.py
```

The pipeline auto-detects ChromaDB and uses semantic search. If ChromaDB isn't built yet, it falls back to TF-IDF automatically.

---

## LangGraph Studio

Visualise and debug the full 8-node workflow graph:

```bash
uv run langgraph dev
```

Opens a browser UI at `http://localhost:8123`. The `langgraph.json` config points to `src/graph/workflow.py:graph`.

---

## Folder Structure

```
REDROB/
├── main.py                          # Pipeline entry point (CLI)
├── app.py                           # Streamlit UI (recruiter dashboard)
├── embed_candidates.py              # One-time ChromaDB index builder
├── langgraph.json                   # LangGraph Studio config
├── pyproject.toml                   # uv project + dependencies
├── .env.example                     # API key template
├── .streamlit/
│   └── config.toml                  # Streamlit theme (white + aqua green)
├── README.md
├── src/
│   ├── data/
│   │   ├── loader.py                # JSONL + JSON array loader (100K)
│   │   └── jd_reader.py             # .docx / .txt JD reader
│   ├── models/
│   │   ├── candidate.py             # Pydantic: Candidate, Skills, Signals
│   │   ├── job_description.py       # Pydantic: JobDescription, ParsedJD
│   │   └── scoring.py               # Pydantic: CandidateScore, RankingResult
│   ├── pipeline/
│   │   ├── jd_parser.py             # Groq LLM → structured JD requirements
│   │   ├── embedding_filter.py      # ChromaDB semantic search (+ TF-IDF fallback)
│   │   ├── signal_scorer.py         # Redrob behavioral signal scoring
│   │   ├── hybrid_ranker.py         # Semantic + signal + skill match fusion
│   │   ├── llm_reranker.py          # Groq LLM batch reranking with reasoning
│   │   └── output_writer.py         # Submission CSV writer + validator
│   ├── graph/
│   │   ├── state.py                 # LangGraph TypedDict state
│   │   ├── nodes.py                 # 8 pipeline node functions
│   │   └── workflow.py              # StateGraph wiring + `graph` export
│   └── utils/
│       ├── text_utils.py            # Candidate/JD text builders for embedding
│       └── logging_utils.py         # Structured logger
├── cache/
│   ├── chroma_db/                   # ChromaDB semantic index (100K × 384-dim)
│   ├── tfidf_vectorizer.pkl         # TF-IDF fallback (auto-built)
│   └── tfidf_matrix.npz             # TF-IDF matrix (auto-built)
└── output/
    └── submission.csv               # Final ranked output (100 rows)
```

---

## Scoring Breakdown

### Signal Score Components (30% of hybrid)

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| Engagement | 25% | Recruiter response rate + avg response time |
| Reliability | 20% | Interview completion rate + offer acceptance rate |
| Availability | 15% | Open-to-work flag + notice period length |
| Profile Quality | 15% | Completeness score + verified email/phone/LinkedIn |
| Activity | 15% | Days since last active + search appearances + saved by recruiters |
| Skill Validation | 10% | Platform assessment scores + GitHub activity score |

### Score Fusion Formula

```
hybrid_score = 0.45 × semantic_score
             + 0.30 × signal_score
             + 0.25 × skill_match
             × experience_gate           ← multiplier: 1.0 if in range, <1 if under/over

final_score  = 0.60 × llm_score
             + 0.40 × hybrid_score
```

---

## Performance (actual benchmark on 100K candidates)

| Node | First Run | Cached Run |
|------|-----------|------------|
| Load JD | ~1s | ~1s |
| Parse JD (Groq LLM) | ~2s | ~2s |
| Load 100K candidates | ~9s | ~9s |
| ChromaDB semantic search | ~18s (model load) + 0.06s query | ~0.06s |
| Signal scoring (500 candidates) | <1s | <1s |
| Hybrid ranking | <1s | <1s |
| LLM reranking (150→100, Groq) | ~200s | ~200s |
| Write submission CSV | <1s | <1s |
| **Total** | **~4 min** | **~4 min** |

> ChromaDB index build (`embed_candidates.py`): one-time ~5 hour run on CPU.
> After that, every pipeline run uses the pre-built index at near-zero cost.

---

## Submission Format

The output `submission.csv` follows the challenge spec exactly:

```
candidate_id,rank,score,reasoning
CAND_0002025,1,0.8234,"Strong ML background with 5.9y production experience..."
CAND_0011687,2,0.8156,"Senior NLP Engineer with expertise in embeddings..."
...
```

- 100 rows (ranks 1–100)
- Scores non-increasing by rank
- Validated by `India_runs_data_and_ai_challenge/validate_submission.py`
