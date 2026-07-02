# import streamlit as st
# import polars as pl
# from src.config import SUBMISSION_PATH

# st.set_page_config(page_title="CascadeRank Sandbox", layout="wide")

# @st.cache_data
# def load_data():
#     if SUBMISSION_PATH.exists():
#         return pl.read_csv(SUBMISSION_PATH).to_pandas()
#     return None

# st.title("CascadeRank AI Recruitment Pipeline")

# df = load_data()

# if df is not None:
#     col1, col2 = st.columns([1, 3])
    
#     with col1:
#         st.metric(label="Final Candidates", value=len(df))
        
#     with col2:
#         search_query = st.text_input("Search by Candidate ID:")

#     if search_query:
#         df = df[df["candidate_id"].str.contains(search_query, case=False, na=False)]

#     st.dataframe(
#         df,
#         use_container_width=True,
#         hide_index=True,
#         column_config={
#             "rank": st.column_config.NumberColumn("Rank"),
#             "score": st.column_config.NumberColumn("Final Score", format="%.4f"),
#             "candidate_id": st.column_config.TextColumn("Candidate ID"),
#             "reasoning": st.column_config.TextColumn("AI Reasoning")
#         }
#     )
# else:
#     st.error("Submission file not found. Waiting for the pipeline to finish...")
import streamlit as st
import polars as pl
import json
from pathlib import Path

# Import your existing modules
from src.config import SUBMISSION_PATH, ROOT
from src.generator import ReasoningGenerator
from src.retrieval import load_job_description

JD_PATH = ROOT / "data" / "raw" / "job_description.docx"
CANDIDATES_PATH = ROOT / "data" / "raw" / "candidates.jsonl"

st.set_page_config(page_title="CascadeRank Sandbox", layout="wide")

# ════════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS (Prevents reloading data/models on every click)
# ════════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_submission():
    if SUBMISSION_PATH.exists():
        return pl.read_csv(SUBMISSION_PATH).to_pandas()
    return None

@st.cache_data
def load_jd():
    return load_job_description(JD_PATH)

@st.cache_resource
def load_llm():
    # Kept at 1024 to prevent memory leaks and gibberish
    return ReasoningGenerator(n_ctx=1024, n_threads=4)

def get_candidate_raw_data(target_id):
    """Fast scan of the JSONL to grab just the selected candidate's raw profile."""
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if target_id in line:
                candidate = json.loads(line)
                if candidate["candidate_id"] == target_id:
                    return candidate
    return None

# ════════════════════════════════════════════════════════════════════════════════
# UI LAYOUT
# ════════════════════════════════════════════════════════════════════════════════
st.title("CascadeRank AI Recruitment Pipeline")

df = load_submission()

if df is not None:
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader(f"🏆 Top {len(df)} Candidates")
        st.dataframe(
            df[["rank", "score", "candidate_id"]], # <-- Removed "reasoning"
            use_container_width=True,
            hide_index=True
        )
        
    with col2:
        st.subheader("🧠 Real-Time AI Reasoning")
        st.info("Select a candidate to generate live, local LLM reasoning.")
        
        # Dropdown to select a candidate ID
        selected_id = st.selectbox("Select Candidate ID:", df["candidate_id"].tolist())
        
        if st.button("Generate Live AI Analysis", type="primary"):
            with st.spinner(f"Booting Phi-3 and analyzing {selected_id}..."):
                
                # 1. Load the model and JD
                llm = load_llm()
                jd_text = load_jd()
                
                # 2. Fetch the raw profile directly from the dataset
                raw_data = get_candidate_raw_data(selected_id)
                
                if raw_data:
                    # 3. Aggressive Whitelisting (The ultimate context saver)
                    # We build a tiny, strict dictionary to guarantee we stay under 1024 tokens
                    safe_candidate = {
                        "skills": raw_data.get("skills", [])[:5],
                        "yoe": raw_data.get("years_of_experience", "N/A")
                    }
                    
                    # Safely grab and hard-cap the most recent job description
                    if raw_data.get("career_history") and len(raw_data["career_history"]) > 0:
                        recent_job = raw_data["career_history"][0]
                        desc = recent_job.get("description", "")
                        # Hard cap the text to 300 characters
                        safe_candidate["recent_job"] = desc[:300] + "..." if len(desc) > 300 else desc
                        
                    record = {"candidate_id": selected_id, "candidate": safe_candidate}
                    
                    # Hard-cap the Job Description in case it's a massive document
                    safe_jd = jd_text[:1500] + "..." if len(jd_text) > 1500 else jd_text
                    
                    # 4. Generate!
                    result = llm.generate(record, safe_jd)
                    
                    st.success("Analysis Complete!")
                    st.markdown("### AI Verdict:")
                    st.write(result["reasoning"])
                else:
                    st.error("Could not find candidate data.")
else:
    st.error("Submission file not found. Please run the pipeline (Stages 1-3) first.")