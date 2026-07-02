"""
REDROB — AI Candidate Ranking System
Streamlit UI  |  White + Aqua Green theme
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="REDROB · AI Recruiter",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme CSS ──────────────────────────────────────────────────────────────
AQUA   = "#00C9B1"
AQUA_D = "#00A896"
AQUA_L = "#E0FAF7"
WHITE  = "#FFFFFF"
GRAY_L = "#F4F7F9"
GRAY   = "#8A9BB0"
DARK   = "#1A2B3C"

CSS = f"""
<style>
/* ── Global ── */
html, body, [class*="css"] {{
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: {DARK};
}}
.stApp {{ background: {WHITE}; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {DARK} 0%, #0F1E2E 100%);
    border-right: 3px solid {AQUA};
}}
[data-testid="stSidebar"] * {{ color: {WHITE} !important; }}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {{ color: {AQUA} !important; font-weight:600; }}

/* ── Header banner ── */
.redrob-header {{
    background: linear-gradient(135deg, {DARK} 0%, #0F2D40 60%, #004D45 100%);
    padding: 2.2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid {AQUA}44;
    box-shadow: 0 8px 32px rgba(0,201,177,0.15);
}}
.redrob-header h1 {{ color:{WHITE}; font-size:2.4rem; font-weight:800; margin:0; letter-spacing:-0.5px; }}
.redrob-header p  {{ color:{AQUA}; margin:0.3rem 0 0; font-size:1.05rem; }}

/* ── Metric cards ── */
.metric-card {{
    background: {WHITE};
    border: 1.5px solid {AQUA}55;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,201,177,0.08);
    transition: transform .2s, box-shadow .2s;
}}
.metric-card:hover {{ transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0,201,177,0.18); }}
.metric-card .val  {{ font-size:2.2rem; font-weight:800; color:{AQUA_D}; line-height:1; }}
.metric-card .lbl  {{ font-size:.82rem; color:{GRAY}; margin-top:.3rem; font-weight:500; text-transform:uppercase; letter-spacing:.5px; }}

/* ── Rank badge ── */
.rank-badge {{
    display:inline-block; width:36px; height:36px; border-radius:50%;
    background: linear-gradient(135deg,{AQUA},{AQUA_D});
    color:{WHITE}; font-weight:800; font-size:.9rem;
    line-height:36px; text-align:center; box-shadow:0 2px 8px rgba(0,201,177,.35);
}}
.rank-badge.gold   {{ background:linear-gradient(135deg,#FFD700,#FFA500); }}
.rank-badge.silver {{ background:linear-gradient(135deg,#C0C0C0,#A0A0A0); }}
.rank-badge.bronze {{ background:linear-gradient(135deg,#CD7F32,#A0522D); }}

/* ── Score bar ── */
.score-bar-wrap {{ background:{GRAY_L}; border-radius:99px; height:8px; width:100%; margin:.3rem 0; }}
.score-bar      {{ height:8px; border-radius:99px;
                   background:linear-gradient(90deg,{AQUA},{AQUA_D}); }}

/* ── Candidate card ── */
.cand-card {{
    background:{WHITE}; border:1.5px solid #E8F0FE; border-radius:14px;
    padding:1.2rem 1.4rem; margin-bottom:.9rem;
    box-shadow:0 1px 6px rgba(0,0,0,.05);
    transition: border-color .2s, box-shadow .2s;
}}
.cand-card:hover {{ border-color:{AQUA}88; box-shadow:0 4px 16px rgba(0,201,177,.12); }}

/* ── Tags ── */
.skill-tag {{
    display:inline-block; background:{AQUA_L}; color:{AQUA_D};
    border:1px solid {AQUA}66; border-radius:99px;
    padding:.18rem .7rem; font-size:.75rem; font-weight:600; margin:.15rem;
}}
.signal-tag {{
    display:inline-block; background:#FFF3E0; color:#E65100;
    border:1px solid #FFCC8055; border-radius:99px;
    padding:.18rem .7rem; font-size:.75rem; font-weight:600; margin:.15rem;
}}

/* ── Status pills ── */
.pill-green {{ background:#E8F5E9; color:#2E7D32; border-radius:99px; padding:.2rem .8rem; font-size:.8rem; font-weight:700; }}
.pill-gray  {{ background:{GRAY_L}; color:{GRAY}; border-radius:99px; padding:.2rem .8rem; font-size:.8rem; font-weight:700; }}

/* ── Buttons ── */
.stButton > button {{
    background:linear-gradient(135deg,{AQUA},{AQUA_D}) !important;
    color:{WHITE} !important; border:none !important; border-radius:10px !important;
    padding:.6rem 1.8rem !important; font-weight:700 !important; font-size:1rem !important;
    box-shadow:0 4px 14px rgba(0,201,177,.4) !important;
    transition: transform .15s, box-shadow .15s !important;
}}
.stButton > button:hover {{ transform:translateY(-2px) !important; box-shadow:0 6px 20px rgba(0,201,177,.55) !important; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{ border-bottom:2px solid {AQUA_L}; gap:.5rem; }}
.stTabs [data-baseweb="tab"]      {{ border-radius:10px 10px 0 0; padding:.5rem 1.2rem; font-weight:600; }}
.stTabs [aria-selected="true"]    {{ background:{AQUA_L}; color:{AQUA_D} !important; border-bottom:3px solid {AQUA}; }}

/* ── Divider ── */
hr {{ border:none; border-top:1.5px solid {AQUA_L}; margin:1.2rem 0; }}

/* ── Progress / spinner ── */
.stProgress > div > div {{ background:linear-gradient(90deg,{AQUA},{AQUA_D}) !important; border-radius:99px; }}

/* ── Input fields ── */
.stTextInput input, .stNumberInput input {{
    border:1.5px solid {AQUA}44 !important; border-radius:8px !important;
}}
.stTextInput input:focus, .stNumberInput input:focus {{
    border-color:{AQUA} !important; box-shadow:0 0 0 3px {AQUA}22 !important;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "India_runs_data_and_ai_challenge"
CACHE_DIR = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "output"

def _chroma_ready() -> bool:
    return (CACHE_DIR / "chroma_db" / "chroma.sqlite3").exists()

def _tfidf_ready() -> bool:
    return (CACHE_DIR / "tfidf_vectorizer.pkl").exists()

def _submission_exists() -> bool:
    return (OUTPUT_DIR / "submission.csv").exists()

def score_color(score: float) -> str:
    if score >= 0.75: return AQUA_D
    if score >= 0.55: return "#0288D1"
    if score >= 0.35: return "#F57C00"
    return "#C62828"

def rank_badge(rank: int) -> str:
    cls = "gold" if rank == 1 else "silver" if rank == 2 else "bronze" if rank == 3 else ""
    return f'<span class="rank-badge {cls}">{rank}</span>'

def score_bar(score: float) -> str:
    pct = int(score * 100)
    return f'''<div class="score-bar-wrap"><div class="score-bar" style="width:{pct}%"></div></div>'''

def pill(text: str, green: bool = True) -> str:
    if green is True or green == "#E8F5E9":
        return f'<span class="pill-green">{text}</span>'
    return f'<span class="pill-gray">{text}</span>'

@st.cache_data(ttl=30)
def load_submission() -> Optional[pd.DataFrame]:
    p = OUTPUT_DIR / "submission.csv"
    if not p.exists(): return None
    return pd.read_csv(p)

@st.cache_data(ttl=300)
def load_sample_candidates(n: int = 200) -> list:
    p = DATA_DIR / "candidates.jsonl"
    if not p.exists():
        p = DATA_DIR / "sample_candidates.json"
        if not p.exists(): return []
        with open(p) as f: return json.load(f)[:n]
    records = []
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n: break
            line = line.strip()
            if line: records.append(json.loads(line))
    return records

def get_candidate_by_id(cid: str, records: list) -> Optional[dict]:
    for r in records:
        if r.get("candidate_id") == cid: return r
    return None

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding:1rem 0 1.5rem;">
      <div style="font-size:2.8rem;">🎯</div>
      <div style="font-size:1.3rem; font-weight:800; color:{WHITE}; letter-spacing:-0.5px;">REDROB</div>
      <div style="font-size:.78rem; color:{AQUA}; margin-top:.2rem;">AI Candidate Ranking</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='color:{AQUA};font-weight:700;font-size:.8rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:.5rem;'>⚙️ Pipeline Config</div>", unsafe_allow_html=True)

    groq_key = st.text_input("Groq API Key", value=os.getenv("GROQ_API_KEY",""), type="password", help="Your Groq API key from console.groq.com")
    top_k    = st.slider("Embedding Retrieval (top-K)", 100, 1000, 500, 50, help="Candidates retrieved from ChromaDB")
    top_n    = st.slider("LLM Reranker (top-N)", 50, 200, 150, 10, help="Top candidates sent to LLM for reranking")

    st.markdown("---")
    st.markdown(f"<div style='color:{AQUA};font-weight:700;font-size:.8rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:.5rem;'>📊 System Status</div>", unsafe_allow_html=True)

    chroma_ok = _chroma_ready()
    tfidf_ok  = _tfidf_ready()
    sub_ok    = _submission_exists()

    st.markdown(f"""
    <div style="display:flex;flex-direction:column;gap:.5rem;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:.85rem;">Semantic Index</span>
        {pill('✓ ChromaDB Ready') if chroma_ok else pill('⚠ Not built', False)}
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:.85rem;">Submission CSV</span>
        {pill('✓ Ready') if sub_ok else pill('— None', False)}
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not chroma_ok:
        st.warning("Run `embed_candidates.py` to build the semantic index for best results.", icon="💡")

    st.markdown("---")
    st.markdown(f"<div style='color:{GRAY};font-size:.75rem;text-align:center;'>Powered by Groq · ChromaDB · LangGraph</div>", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="redrob-header">
  <h1>🎯 REDROB <span style="color:{AQUA};font-weight:400;">AI Recruiter</span></h1>
  <p>Rank 100,000 candidates the way a great recruiter would — semantically, intelligently, explainably.</p>
</div>
""", unsafe_allow_html=True)

# ── Quick stats row ─────────────────────────────────────────────────────────
df = load_submission()
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="val">100K</div><div class="lbl">Candidates Indexed</div></div>', unsafe_allow_html=True)
with c2:
    n_ranked = len(df) if df is not None else 0
    st.markdown(f'<div class="metric-card"><div class="val">{n_ranked}</div><div class="lbl">Ranked in Last Run</div></div>', unsafe_allow_html=True)
with c3:
    backend = "ChromaDB 🔷" if chroma_ok else ("TF-IDF ⚡" if tfidf_ok else "None ⚠️")
    st.markdown(f'<div class="metric-card"><div class="val" style="font-size:1.3rem;">{backend}</div><div class="lbl">Retrieval Backend</div></div>', unsafe_allow_html=True)
with c4:
    top_score = f"{df['score'].max():.4f}" if df is not None else "—"
    st.markdown(f'<div class="metric-card"><div class="val">{top_score}</div><div class="lbl">Top Candidate Score</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────────────────
tab_run, tab_results, tab_analytics, tab_candidate = st.tabs([
    "🚀  Run Pipeline", "🏆  Rankings", "📊  Analytics", "🔍  Candidate Deep Dive"
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — Run Pipeline
# ══════════════════════════════════════════════════════════════════════
with tab_run:
    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown(f"### ⚙️ Pipeline Settings")

        # ── PDF upload drop zone ─────────────────────────────────────
        st.markdown(f"""
        <div style="font-size:.85rem;font-weight:600;color:{DARK};margin-bottom:.4rem;">
          📄 Job Description
        </div>
        """, unsafe_allow_html=True)

        uploaded_pdf = st.file_uploader(
            "Drop your JD PDF here or click to browse",
            type=["pdf"],
            help="Upload a PDF of the job description. Leave empty to use the default job_description.docx.",
            label_visibility="visible",
        )

        custom_jd = ""
        jd_path_override = None

        if uploaded_pdf is not None:
            # Extract text from PDF
            try:
                import PyPDF2, io
                reader = PyPDF2.PdfReader(io.BytesIO(uploaded_pdf.read()))
                pdf_text = "\n".join(
                    page.extract_text() or "" for page in reader.pages
                ).strip()
                if pdf_text:
                    custom_jd = pdf_text
                    tmp_jd = BASE_DIR / "output" / "_uploaded_jd.txt"
                    OUTPUT_DIR.mkdir(exist_ok=True)
                    tmp_jd.write_text(pdf_text, encoding="utf-8")
                    jd_path_override = str(tmp_jd)
                    st.markdown(f"""
                    <div style="background:{AQUA_L};border:1.5px solid {AQUA}66;border-radius:10px;
                                padding:.7rem 1rem;font-size:.83rem;color:{AQUA_D};margin-top:.4rem;">
                      ✅ <b>{uploaded_pdf.name}</b> — {len(pdf_text):,} chars extracted from {len(reader.pages)} page(s)
                    </div>""", unsafe_allow_html=True)
                else:
                    st.warning("Could not extract text from PDF. Using default JD.")
            except Exception as e:
                st.warning(f"PDF read error: {e}. Using default JD.")
        else:
            st.markdown(f"""
            <div style="background:{GRAY_L};border:1.5px dashed {GRAY}55;border-radius:10px;
                        padding:.55rem 1rem;font-size:.8rem;color:{GRAY};margin-top:.2rem;">
              Using default: <b>job_description.docx</b>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Output**")
        output_name = st.text_input("Output filename", value="submission.csv", label_visibility="collapsed")
        rebuild = False  # handled internally

        run_btn = st.button("🚀  Run Ranking Pipeline", use_container_width=True)

    with col_right:
        st.markdown(f"### 🗺️ Pipeline Flow")
        steps = [
            ("1", "Load JD", "Reads job description file"),
            ("2", "Parse JD", "Groq LLM extracts requirements"),
            ("3", "Load Candidates", "Streams 100K JSONL records"),
            ("4", "Semantic Search", "ChromaDB → top-500 matches"),
            ("5", "Signal Scorer", "6 behavioral signal components"),
            ("6", "Hybrid Ranker", "Semantic + signals + skill match"),
            ("7", "LLM Reranker", "Groq llama-3.3-70b deep ranking"),
            ("8", "Write Output", "Validated submission.csv"),
        ]
        for num, name, desc in steps:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:.8rem;padding:.45rem .7rem;
                        border-radius:10px;margin:.3rem 0;background:{AQUA_L};">
              <div style="background:{AQUA};color:{WHITE};border-radius:50%;
                          width:26px;height:26px;text-align:center;line-height:26px;
                          font-weight:800;font-size:.8rem;flex-shrink:0;">{num}</div>
              <div>
                <div style="font-weight:700;font-size:.88rem;color:{DARK};">{name}</div>
                <div style="font-size:.75rem;color:{GRAY};">{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    # ── Run logic ────────────────────────────────────────────────────
    if run_btn:
        if not groq_key:
            st.error("Please enter your Groq API key in the sidebar.")
        else:
            log_area    = st.empty()
            progress    = st.progress(0, text="Initialising pipeline...")
            status_area = st.empty()

            log_lines: list[str] = []

            def append_log(msg: str):
                log_lines.append(msg)
                log_area.code("\n".join(log_lines[-30:]), language="")

            try:
                from src.graph.workflow import build_ranking_graph
                from src.models.job_description import JobDescription

                append_log("⚡ Building pipeline graph...")
                progress.progress(5, "Building LangGraph pipeline...")
                graph = build_ranking_graph()

                jd_path = str(DATA_DIR / "job_description.docx")

                # Use uploaded PDF path if provided
                if jd_path_override:
                    jd_path = jd_path_override
                    append_log(f"📄 Using uploaded PDF JD ({len(custom_jd)} chars)")

                out_path = str(OUTPUT_DIR / output_name)
                OUTPUT_DIR.mkdir(exist_ok=True)

                node_progress = {
                    "load_jd": 10, "parse_jd": 20, "load_candidates": 35,
                    "embedding_filter": 55, "signal_scorer": 65,
                    "hybrid_ranker": 75, "llm_reranker": 92, "write_output": 100,
                }
                node_labels = {
                    "load_jd": "Loading JD...", "parse_jd": "Parsing JD with LLM...",
                    "load_candidates": "Loading 100K candidates...",
                    "embedding_filter": "Semantic search (ChromaDB)...",
                    "signal_scorer": "Scoring behavioral signals...",
                    "hybrid_ranker": "Hybrid ranking...",
                    "llm_reranker": "LLM reranking (Groq)...",
                    "write_output": "Writing submission...",
                }

                initial_state = {
                    "jd_path": jd_path,
                    "candidates_path": str(DATA_DIR / "candidates.jsonl"),
                    "output_path": out_path,
                    "groq_api_key": groq_key,
                    "cache_dir": str(CACHE_DIR),
                    "top_k_embed": top_k,
                    "top_n_hybrid": top_n,
                    "top_n_final": 100,
                    "rebuild_index": rebuild,
                    "raw_jd": None, "parsed_jd": None,
                    "all_candidates": None, "semantic_results": None,
                    "signal_scores": None, "hybrid_scores": None,
                    "final_scores": None, "submission_path": None,
                    "errors": [], "status": "starting", "node_timings": {},
                }

                append_log("🔄 Pipeline running — this takes ~4 minutes...")
                t_start = time.time()

                # Stream node-by-node for live progress
                for event in graph.stream(initial_state):
                    for node_name in event:
                        pct   = node_progress.get(node_name, 50)
                        label = node_labels.get(node_name, node_name)
                        progress.progress(pct, label)
                        timings = event[node_name].get("node_timings") or {}
                        elapsed = timings.get(node_name, 0)
                        append_log(f"✅ {node_name:<22} {elapsed:>5.1f}s")

                elapsed_total = round(time.time() - t_start, 1)
                progress.progress(100, "Complete!")
                append_log(f"\n🎉 Done in {elapsed_total}s")

                st.success(f"✅ Pipeline complete in {elapsed_total}s — {out_path}")
                st.cache_data.clear()
                st.rerun()

            except Exception as e:
                st.error(f"Pipeline error: {e}")
                append_log(f"❌ Error: {e}")

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — Rankings
# ══════════════════════════════════════════════════════════════════════
with tab_results:
    if df is None:
        st.markdown(f"""
        <div style="text-align:center;padding:4rem 2rem;background:{AQUA_L};border-radius:16px;border:2px dashed {AQUA};">
          <div style="font-size:3rem;">🏆</div>
          <div style="font-size:1.3rem;font-weight:700;color:{DARK};margin:.5rem 0;">No rankings yet</div>
          <div style="color:{GRAY};">Run the pipeline in the <b>Run Pipeline</b> tab to generate results.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Controls row
        fc1, fc2, fc3 = st.columns([2, 1, 1])
        with fc1:
            search_q = st.text_input("🔍 Search candidates", placeholder="Filter by ID or reasoning keywords...")
        with fc2:
            show_n = st.selectbox("Show", [10, 25, 50, 100], index=1)
        with fc3:
            min_score = st.slider("Min score", 0.0, 1.0, 0.0, 0.05)

        filtered = df.copy()
        if search_q:
            mask = (
                filtered["candidate_id"].str.contains(search_q, case=False, na=False) |
                filtered["reasoning"].str.contains(search_q, case=False, na=False)
            )
            filtered = filtered[mask]
        filtered = filtered[filtered["score"] >= min_score].head(show_n)

        st.markdown(f"<div style='color:{GRAY};font-size:.85rem;margin-bottom:.8rem;'>Showing {len(filtered)} candidates</div>", unsafe_allow_html=True)

        # Download button
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download submission.csv", csv_bytes, "submission.csv", "text/csv", use_container_width=False)

        st.markdown("<br>", unsafe_allow_html=True)

        # Cards
        for _, row in filtered.iterrows():
            rank  = int(row["rank"])
            score = float(row["score"])
            cid   = row["candidate_id"]
            rsn   = str(row["reasoning"])
            pct   = int(score * 100)
            color = score_color(score)

            st.markdown(f"""
            <div class="cand-card">
              <div style="display:flex;align-items:center;gap:1rem;margin-bottom:.6rem;">
                {rank_badge(rank)}
                <div style="flex:1;">
                  <div style="font-weight:800;font-size:1rem;color:{DARK};">{cid}</div>
                  <div style="font-size:.78rem;color:{GRAY};">Rank #{rank}</div>
                </div>
                <div style="text-align:right;">
                  <div style="font-size:1.6rem;font-weight:800;color:{color};">{score:.4f}</div>
                  <div style="font-size:.72rem;color:{GRAY};">Score</div>
                </div>
              </div>
              {score_bar(score)}
              <div style="font-size:.83rem;color:{GRAY};margin-top:.5rem;line-height:1.5;">
                💬 {rsn}
              </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — Analytics
# ══════════════════════════════════════════════════════════════════════
with tab_analytics:
    if df is None:
        st.info("Run the pipeline first to see analytics.")
    else:
        ac1, ac2 = st.columns(2)

        # Score distribution
        with ac1:
            st.markdown(f"#### Score Distribution")
            fig = px.histogram(
                df, x="score", nbins=20,
                color_discrete_sequence=[AQUA],
                labels={"score": "Final Score", "count": "Count"},
                template="simple_white",
            )
            fig.update_layout(
                plot_bgcolor=WHITE, paper_bgcolor=WHITE,
                font=dict(family="Inter, sans-serif", color=DARK),
                margin=dict(l=10, r=10, t=30, b=10),
                bargap=0.08,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor=AQUA_L),
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

        # Score by rank (top 20)
        with ac2:
            st.markdown(f"#### Score Decay Curve (Top 50)")
            top50 = df.head(50)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=top50["rank"], y=top50["score"],
                mode="lines+markers",
                line=dict(color=AQUA, width=2.5),
                marker=dict(color=AQUA_D, size=6, line=dict(color=WHITE, width=1.5)),
                fill="tozeroy", fillcolor=f"rgba(0,201,177,0.08)",
                name="Score",
            ))
            fig2.update_layout(
                plot_bgcolor=WHITE, paper_bgcolor=WHITE,
                font=dict(family="Inter, sans-serif", color=DARK),
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(title="Rank", showgrid=False),
                yaxis=dict(title="Score", showgrid=True, gridcolor=AQUA_L),
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Score bands summary
        st.markdown(f"#### Score Band Breakdown")
        bands = {
            "🟢 Exceptional (≥0.75)": len(df[df["score"] >= 0.75]),
            "🔵 Strong (0.55–0.74)":  len(df[(df["score"] >= 0.55) & (df["score"] < 0.75)]),
            "🟠 Moderate (0.35–0.54)": len(df[(df["score"] >= 0.35) & (df["score"] < 0.55)]),
            "🔴 Weak (<0.35)":         len(df[df["score"] < 0.35]),
        }
        bc1, bc2, bc3, bc4 = st.columns(4)
        for col, (label, count) in zip([bc1, bc2, bc3, bc4], bands.items()):
            with col:
                pct = f"{count/len(df)*100:.0f}%"
                st.markdown(f'<div class="metric-card"><div class="val">{count}</div><div class="lbl">{label}<br><span style="color:{AQUA_D};font-size:.9rem;">{pct}</span></div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Top 10 table
        st.markdown(f"#### Top 10 Ranked Candidates")
        top10 = df.head(10)[["rank", "candidate_id", "score", "reasoning"]].copy()
        top10["score"] = top10["score"].map("{:.4f}".format)
        st.dataframe(
            top10,
            use_container_width=True,
            hide_index=True,
            column_config={
                "rank":         st.column_config.NumberColumn("Rank", width=70),
                "candidate_id": st.column_config.TextColumn("Candidate ID", width=140),
                "score":        st.column_config.TextColumn("Score", width=90),
                "reasoning":    st.column_config.TextColumn("Reasoning"),
            }
        )

# ══════════════════════════════════════════════════════════════════════
# TAB 4 — Candidate Deep Dive
# ══════════════════════════════════════════════════════════════════════
with tab_candidate:
    st.markdown("#### 🔍 Candidate Deep Dive")

    dd1, dd2 = st.columns([1, 2])
    with dd1:
        if df is not None:
            cid_options = df["candidate_id"].tolist()
            selected_cid = st.selectbox("Select ranked candidate", cid_options,
                format_func=lambda x: f"{x}  (rank #{df[df['candidate_id']==x]['rank'].values[0]})")
        else:
            selected_cid = st.text_input("Enter Candidate ID", placeholder="CAND_0000001")

    records = load_sample_candidates(500)
    rec = get_candidate_by_id(selected_cid, records) if selected_cid else None

    if rec is None and selected_cid:
        # Try loading from full jsonl
        try:
            with open(DATA_DIR / "candidates.jsonl", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line.strip())
                    if d.get("candidate_id") == selected_cid:
                        rec = d
                        break
        except Exception:
            pass

    if rec:
        profile  = rec.get("profile", {})
        signals  = rec.get("redrob_signals", {})
        skills   = rec.get("skills", [])
        career   = rec.get("career_history", [])
        education= rec.get("education", [])
        certs    = rec.get("certifications", [])

        # Header
        rank_row = df[df["candidate_id"] == selected_cid] if df is not None else None
        rank_txt = f"Rank #{int(rank_row['rank'].values[0])}" if rank_row is not None and len(rank_row) else ""
        score_txt= f"{float(rank_row['score'].values[0]):.4f}" if rank_row is not None and len(rank_row) else ""
        reasoning= rank_row['reasoning'].values[0] if rank_row is not None and len(rank_row) else ""

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{DARK},{AQUA_D}88);border-radius:14px;padding:1.4rem 1.6rem;margin-bottom:1rem;">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
              <div style="font-size:1.3rem;font-weight:800;color:{WHITE};">{profile.get('anonymized_name','—')}</div>
              <div style="color:{AQUA};font-size:.9rem;margin-top:.2rem;">{profile.get('headline','')}</div>
              <div style="color:#CBD5E1;font-size:.82rem;margin-top:.4rem;">
                📍 {profile.get('location','')}, {profile.get('country','')} &nbsp;·&nbsp;
                🏢 {profile.get('current_company','')} &nbsp;·&nbsp;
                ⏱️ {profile.get('years_of_experience',0)} yrs exp
              </div>
            </div>
            <div style="text-align:right;">
              <div style="font-size:2rem;font-weight:800;color:{AQUA};">{score_txt}</div>
              <div style="color:{WHITE};font-size:.85rem;">{rank_txt}</div>
            </div>
          </div>
          {f'<div style="margin-top:.9rem;background:rgba(255,255,255,.1);border-radius:8px;padding:.6rem .9rem;color:#CBD5E1;font-size:.82rem;">💬 {reasoning}</div>' if reasoning else ''}
        </div>
        """, unsafe_allow_html=True)

        # Details columns
        col_a, col_b = st.columns(2)

        with col_a:
            # Skills
            st.markdown(f"**🛠 Skills**")
            top_skills = sorted(skills, key=lambda s: {"expert":4,"advanced":3,"intermediate":2,"beginner":1}.get(s.get("proficiency",""),0), reverse=True)
            skill_html = "".join(
                f'<span class="skill-tag">{s["name"]} <span style="opacity:.7;font-size:.7rem;">({s.get("proficiency","")[:3].upper()})</span></span>'
                for s in top_skills[:15]
            )
            st.markdown(skill_html or "<em>None listed</em>", unsafe_allow_html=True)

            # Education
            st.markdown(f"<br>**🎓 Education**", unsafe_allow_html=True)
            for e in education:
                st.markdown(f"""
                <div style="background:{GRAY_L};border-radius:8px;padding:.6rem .9rem;margin:.3rem 0;font-size:.83rem;">
                  <b>{e.get('degree','')} in {e.get('field_of_study','')}</b><br>
                  <span style="color:{GRAY};">{e.get('institution','')} · {e.get('end_year','')} · {e.get('tier','').replace('_',' ').title()}</span>
                </div>""", unsafe_allow_html=True)

            # Certs
            if certs:
                st.markdown(f"<br>**📜 Certifications**", unsafe_allow_html=True)
                for c in certs:
                    st.markdown(f'<span class="skill-tag">🏅 {c.get("name",str(c))}</span>', unsafe_allow_html=True)

        with col_b:
            # Platform signals radar
            st.markdown("**📡 Platform Signals**")
            sig_vals = {
                "Response Rate":    signals.get("recruiter_response_rate", 0),
                "Interview Rate":   signals.get("interview_completion_rate", 0),
                "Profile Complete": signals.get("profile_completeness_score", 0) / 100,
                "GitHub Activity":  max(0, signals.get("github_activity_score", 0)) / 100,
                "Offer Acceptance": max(0, signals.get("offer_acceptance_rate", 0)),
            }
            categories = list(sig_vals.keys())
            values     = list(sig_vals.values()) + [list(sig_vals.values())[0]]

            fig_r = go.Figure(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor=f"rgba(0,201,177,0.18)",
                line=dict(color=AQUA, width=2),
                marker=dict(color=AQUA_D, size=6),
            ))
            fig_r.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9), gridcolor=AQUA_L),
                    angularaxis=dict(tickfont=dict(size=9, color=DARK)),
                    bgcolor=WHITE,
                ),
                paper_bgcolor=WHITE,
                margin=dict(l=30, r=30, t=30, b=10),
                showlegend=False,
                height=260,
            )
            st.plotly_chart(fig_r, use_container_width=True)

            # Quick signal tags
            tags = []
            if signals.get("open_to_work_flag"): tags.append(("✅ Open to Work", True))
            if signals.get("willing_to_relocate"): tags.append(("✈️ Will Relocate", True))
            if signals.get("verified_email"): tags.append(("📧 Verified Email", True))
            if signals.get("linkedin_connected"): tags.append(("🔗 LinkedIn", True))
            nd = signals.get("notice_period_days", 90)
            tags.append((f"📅 {nd}d Notice", nd <= 30))
            tag_html = "".join(f'<span class="{"skill-tag" if ok else "signal-tag"}">{t}</span>' for t, ok in tags)
            st.markdown(tag_html, unsafe_allow_html=True)

        # Career history
        st.markdown("<br>**💼 Career History**", unsafe_allow_html=True)
        for job in career:
            current_badge = ' <span style="background:#E8F5E9;color:#2E7D32;border-radius:4px;padding:.1rem .4rem;font-size:.72rem;font-weight:700;">CURRENT</span>' if job.get("is_current") else ""
            st.markdown(f"""
            <div class="cand-card" style="padding:.9rem 1.1rem;">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                  <div style="font-weight:700;font-size:.92rem;">{job.get('title','')}{current_badge}</div>
                  <div style="color:{GRAY};font-size:.8rem;">{job.get('company','')} · {job.get('industry','')} · {job.get('company_size','')} employees</div>
                </div>
                <div style="color:{AQUA_D};font-size:.8rem;font-weight:600;text-align:right;">
                  {job.get('duration_months',0)} months
                </div>
              </div>
              <div style="font-size:.8rem;color:{GRAY};margin-top:.5rem;line-height:1.5;">{job.get('description','')[:300]}{'...' if len(job.get('description',''))>300 else ''}</div>
            </div>""", unsafe_allow_html=True)

    elif selected_cid:
        st.warning(f"Candidate `{selected_cid}` not found in the loaded sample. Full JSONL search may take a moment.")
    else:
        st.markdown(f"""
        <div style="text-align:center;padding:3rem;background:{AQUA_L};border-radius:14px;border:2px dashed {AQUA};">
          <div style="font-size:2.5rem;">🔍</div>
          <div style="font-size:1.1rem;font-weight:700;color:{DARK};margin:.4rem 0;">Select a candidate</div>
          <div style="color:{GRAY};">Pick from the Rankings tab or enter a Candidate ID above.</div>
        </div>
        """, unsafe_allow_html=True)
