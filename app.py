import streamlit as st
import polars as pl
from src.config import SUBMISSION_PATH

st.set_page_config(page_title="CascadeRank Sandbox", layout="wide")

@st.cache_data
def load_data():
    if SUBMISSION_PATH.exists():
        return pl.read_csv(SUBMISSION_PATH).to_pandas()
    return None

st.title("CascadeRank AI Recruitment Pipeline")

df = load_data()

if df is not None:
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.metric(label="Final Candidates", value=len(df))
        
    with col2:
        search_query = st.text_input("Search by Candidate ID:")

    if search_query:
        df = df[df["candidate_id"].str.contains(search_query, case=False, na=False)]

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "rank": st.column_config.NumberColumn("Rank"),
            "score": st.column_config.NumberColumn("Final Score", format="%.4f"),
            "candidate_id": st.column_config.TextColumn("Candidate ID"),
            "reasoning": st.column_config.TextColumn("AI Reasoning")
        }
    )
else:
    st.error("Submission file not found. Waiting for the pipeline to finish...")