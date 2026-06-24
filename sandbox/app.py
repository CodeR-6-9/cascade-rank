"""
sandbox/app.py — Minimal Streamlit UI for the required sandbox link.

Allows judges to:
  1. Upload a small candidates JSON/JSONL file
  2. Run the full pipeline on it
  3. See the top 20 ranked candidates
  4. Download the submission CSV

Run locally:
    pip3 install streamlit
    streamlit run sandbox/app.py
"""

import json
import sys
import os
import tempfile
import time

import streamlit as st
import polars as pl

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Cascade Rank — Redrob Candidate Ranker",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Cascade Rank")
st.caption("Intelligent Candidate Discovery & Ranking — Redrob Hackathon")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("About")
st.sidebar.markdown("""
**Pipeline stages:**
1. 🔍 Parse & filter (hard disqualifiers)
2. 🧠 Semantic scoring (sentence-transformers + FAISS)
3. 📊 Behavioral signal ranking
4. ✍️ Reasoning generation

**Compute constraints met:**
- ✅ CPU only
- ✅ No network calls during ranking
- ✅ < 16 GB RAM
- ✅ < 5 min runtime
""")

st.sidebar.header("Settings")
use_mock = st.sidebar.checkbox(
    "Use mock semantic scores",
    value=True,
    help="Uncheck once retrieval.py is ready to use real sentence-transformer scores"
)

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------
st.header("1. Upload Candidates")
st.markdown("Upload a `.json` (array) or `.jsonl` (one candidate per line) file.")

uploaded = st.file_uploader(
    "Choose file",
    type=["json", "jsonl"],
    help="Use sample_candidates.json from the hackathon bundle to test"
)

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
if uploaded is not None:
    st.header("2. Run Pipeline")

    if st.button("▶ Run Ranker", type="primary"):
        with st.spinner("Running pipeline..."):

            # Save upload to temp file
            suffix = ".json" if uploaded.name.endswith(".json") else ".jsonl"
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=suffix, delete=False
            ) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            try:
                t0 = time.time()

                # Stage 1 — parse
                from parser import run_parser
                df = run_parser(tmp_path)
                st.success(f"✅ Stage 1: {df.height} candidates passed hard filters")

                # Stage 2 — semantic scoring
                if use_mock:
                    from pipeline import _mock_semantic_scores
                    df = _mock_semantic_scores(df)
                    st.info("ℹ️ Stage 2: Mock semantic scores (enable real scores in Settings)")
                else:
                    try:
                        from retrieval import run_retrieval
                        df = run_retrieval(df)
                        st.success("✅ Stage 2: Real semantic scores applied")
                    except ImportError:
                        from pipeline import _mock_semantic_scores
                        df = _mock_semantic_scores(df)
                        st.warning("⚠️ Stage 2: retrieval.py not found — using mock scores")

                # Stage 3 — rank
                from ranker import rank_candidates, build_submission
                ranked     = rank_candidates(df)
                submission = build_submission(ranked)
                st.success(f"✅ Stage 3: Top {ranked.height} candidates ranked")

                elapsed = time.time() - t0
                st.success(f"⏱️ Total time: {elapsed:.1f}s")

                # Store in session state
                st.session_state["submission"] = submission

            except Exception as e:
                st.error(f"Pipeline error: {e}")
                raise
            finally:
                os.unlink(tmp_path)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if "submission" in st.session_state:
    submission = st.session_state["submission"]
    rows = submission.to_dicts()

    st.header("3. Results")

    # Top 20 table
    st.subheader("Top 20 Candidates")
    display_rows = []
    for r in rows[:20]:
        display_rows.append({
            "Rank"        : r["rank"],
            "Candidate ID": r["candidate_id"],
            "Score"       : round(r["score"], 4),
            "Reasoning"   : r["reasoning"],
        })
    st.dataframe(display_rows, use_container_width=True)

    # Score distribution
    st.subheader("Score Distribution (Top 100)")
    scores = [r["score"] for r in rows]
    st.bar_chart({"Score": scores})

    # Download
    st.header("4. Download Submission")
    csv_bytes = submission.write_csv().encode("utf-8")
    st.download_button(
        label="⬇️ Download submission.csv",
        data=csv_bytes,
        file_name="submission.csv",
        mime="text/csv",
    )

else:
    st.info("Upload a candidates file and click **Run Ranker** to see results.")