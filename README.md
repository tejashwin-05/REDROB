# REDROB — AI Candidate Ranking System

An intelligent candidate ranking pipeline that ranks 100K candidates the way a great recruiter would — using semantic understanding, behavioral signals, and LLM reasoning.

## Architecture

```
Job Description (.docx)
        │
        ▼
┌─────────────────┐
│  1. load_jd     │  Read + parse job description file
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. parse_jd    │  Groq LLM extracts structured requirements
└────────┬────────┘  (skills, experience, responsibilities, disqualifiers)
         │
         ▼
┌──────────────────────┐
│  3. load_candidates  │  Stream 100K JSONL candidates into memory
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  4. embedding_filter │  FAISS + BGE-small: retrieve top-500 semantically
└──────────┬───────────┘  similar candidates (cached to disk)
           │
           ▼
┌──────────────────────┐
│  5. signal_scorer    │  Score Redrob behavioral signals:
└──────────┬───────────┘  engagement, reliability, availability,
           │              profile quality, activity, skill validation
           ▼
┌──────────────────────┐
│  6. hybrid_ranker    │  Combine: 45% semantic + 30% signal + 25% skill match
└──────────┬───────────┘  + experience gate → top-150
           │
           ▼
┌──────────────────────┐
│  7. llm_reranker     │  Groq llama-3.3-70b reranks top-150 in batches of 10
└──────────┬───────────┘  Final = 60% LLM + 40% hybrid
           │
           ▼
┌──────────────────────┐
│  8. write_output     │  Validates + writes submission.csv (100 rows)
└──────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph (StateGraph) |
| LLM | Groq `llama-3.3-70b-versatile` |
| Stage-1 Retrieval | TF-IDF + cosine similarity (scikit-learn, CPU, instant) |
| Stage-2 Rerank (optional) | `BAAI/bge-small-en-v1.5` via fastembed ONNX |
| Fuzzy Skill Match | RapidFuzz |
| Data Models | Pydantic v2 |
| Package Manager | uv |
| Python | 3.14 |

## Quick Start

```bash
# Install dependencies
uv sync

# Run the full pipeline
uv run python main.py

# Force rebuild FAISS index (first run or after data changes)
uv run python main.py --rebuild-index

# Validate output
uv run python India_runs_data_and_ai_challenge/validate_submission.py output/submission.csv
```

## LangGraph Studio

To visualize and debug the workflow graph:

```bash
# Install LangGraph CLI
uv add langgraph-cli

# Launch LangGraph Studio (opens browser)
uv run langgraph dev
```

The `langgraph.json` file points Studio to `src/graph/workflow.py:graph`.

## CLI Options

```
--candidates   Path to candidates.jsonl  [default: India_runs_data_and_ai_challenge/candidates.jsonl]
--jd           Path to job description   [default: India_runs_data_and_ai_challenge/job_description.docx]
--output       Output CSV path           [default: output/submission.csv]
--cache-dir    FAISS cache directory     [default: cache/]
--top-k        Embedding retrieval count [default: 500]
--top-n        Candidates sent to LLM    [default: 150]
--rebuild-index  Force FAISS rebuild     [default: False]
```

## Folder Structure

```
REDROB/
├── main.py                          # Pipeline entry point
├── langgraph.json                   # LangGraph Studio config
├── pyproject.toml                   # uv project config
├── .env                             # GROQ_API_KEY
├── src/
│   ├── data/
│   │   ├── loader.py                # Streaming JSONL loader (100K candidates)
│   │   └── jd_reader.py             # .docx / .txt JD reader
│   ├── models/
│   │   ├── candidate.py             # Pydantic: Candidate, Skills, Signals...
│   │   ├── job_description.py       # Pydantic: JobDescription, ParsedJD
│   │   └── scoring.py               # Pydantic: CandidateScore, RankingResult
│   ├── pipeline/
│   │   ├── jd_parser.py             # Groq LLM JD understanding
│   │   ├── embedding_filter.py      # FAISS semantic retrieval
│   │   ├── signal_scorer.py         # Redrob behavioral signal scoring
│   │   ├── hybrid_ranker.py         # Semantic + signal + skill fusion
│   │   ├── llm_reranker.py          # Groq LLM batch reranking with reasoning
│   │   └── output_writer.py         # Submission CSV writer + validator
│   ├── graph/
│   │   ├── state.py                 # LangGraph TypedDict state
│   │   ├── nodes.py                 # Node functions (8 pipeline steps)
│   │   └── workflow.py              # StateGraph wiring + `graph` export
│   └── utils/
│       ├── text_utils.py            # Candidate/JD text builders for embedding
│       └── logging_utils.py         # Structured logger
├── cache/                           # FAISS index + candidate IDs (auto-generated)
├── output/                          # submission.csv (auto-generated)
└── India_runs_data_and_ai_challenge/
    ├── candidates.jsonl             # 100K candidate profiles
    ├── job_description.docx         # Target role JD
    └── ...
```

## Performance (actual benchmark on 100K candidates)

| Node | Time |
|------|------|
| Load JD | ~1s |
| Parse JD (Groq) | ~5s |
| Load 100K candidates | ~44s |
| TF-IDF index build + search | ~158s |
| Signal scoring (500 candidates) | ~0.1s |
| Hybrid ranking | ~1s |
| LLM reranking (150→100, Groq) | ~191s |
| Write submission CSV | ~0.1s |
| **Total** | **~400s (~7 min)** |

Second run onwards (cached TF-IDF index): **~4 min** (index load in ~18s instead of ~158s).

### Signal Score (30% of hybrid)
| Component | Weight | What it measures |
|-----------|--------|-----------------|
| Engagement | 25% | Recruiter response rate + response speed |
| Reliability | 20% | Interview completion + offer acceptance |
| Availability | 15% | Open to work + notice period |
| Profile Quality | 15% | Completeness + verified contacts |
| Activity | 15% | Recency, search appearances, saved by recruiters |
| Skill Validation | 10% | Assessment scores + GitHub activity |

### Hybrid Score
```
hybrid = 0.45 × semantic_score + 0.30 × signal_score + 0.25 × skill_match
final  = 0.60 × llm_score      + 0.40 × hybrid_score
```

## Scoring Breakdown

### Signal Score (30% of hybrid)
| Component | Weight | What it measures |
|-----------|--------|-----------------|
| Engagement | 25% | Recruiter response rate + response speed |
| Reliability | 20% | Interview completion + offer acceptance |
| Availability | 15% | Open to work + notice period |
| Profile Quality | 15% | Completeness + verified contacts |
| Activity | 15% | Recency, search appearances, saved by recruiters |
| Skill Validation | 10% | Assessment scores + GitHub activity |

### Hybrid Score
```
hybrid = 0.45 × tfidf_score + 0.30 × signal_score + 0.25 × skill_match
final  = 0.60 × llm_score   + 0.40 × hybrid_score
```
