"""
retrieval.py — Stage 2: sentence-transformers + FAISS semantic scoring.
Person B owns the core logic. This file wraps it to integrate with the pipeline.

Two modes:
  1. run_retrieval(df)     — pipeline integration: takes parser.py DataFrame,
                             returns same DataFrame + 'semantic_score' column
  2. build_and_save_index()— standalone: pre-builds FAISS index from full JSONL
                             (run once before the pipeline)

Run index build:
    python -m src.retrieval --build

Run standalone test:
    python -m src.retrieval --sample
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CANDIDATES_JSONL, SAMPLE_CANDIDATES,
    OUTPUT_DIR,
)

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent
ROOT_DIR       = BASE_DIR.parent
ARTIFACTS_DIR  = ROOT_DIR / "artifacts"
MODEL_PATH     = ROOT_DIR / "models" / "all-MiniLM-L6-v2"
JD_PATH        = ROOT_DIR / "data" / "raw" / "job_description.docx"

EXPECTED_DIM   = 384
TOP_K_RETRIEVE = 500   # retrieve top 500 from FAISS, ranker picks final 100


# ── Text builder (B's logic, unchanged) ───────────────────────────────────────

def build_retrieval_document(candidate: dict) -> str:
    profile = candidate.get("profile", {})

    skill_lines = [
        f"- {s.get('name', '')} ({s.get('proficiency', 'Unknown').title()})"
        for s in candidate.get("skills", [])
    ]

    career_entries = [
        f"{j.get('title', '')} | {j.get('company', '')}\n{j.get('description', '')}"
        for j in candidate.get("career_history", [])
    ]

    edu_entries = [
        f"{e.get('degree', '')} in {e.get('field_of_study', '')}"
        for e in candidate.get("education", [])
    ]

    sections = [
        "Current Title:", profile.get("current_title", ""), "",
        "Years of Experience:", str(profile.get("years_of_experience", "")), "",
        "Headline:", profile.get("headline", ""), "",
        "Professional Summary:", profile.get("summary", ""), "",
        "Skills:", "\n".join(skill_lines), "",
        "Career History:", "", "\n\n".join(career_entries), "",
        "Education:", "\n".join(edu_entries),
    ]
    return "\n".join(sections).strip()


# ── JD loader ─────────────────────────────────────────────────────────────────

def load_job_description(path: Path = JD_PATH) -> str:
    try:
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        log.warning("Could not load JD from docx (%s) — using fallback text", e)
        return (
            "Senior AI Engineer with experience in embeddings, vector search, "
            "FAISS, sentence-transformers, NLP, retrieval systems, ranking, "
            "Python, PyTorch, production ML deployment, evaluation frameworks."
        )


# ── Model loader ───────────────────────────────────────────────────────────────

def _load_model():
    from sentence_transformers import SentenceTransformer
    # Use local model if available, else download
    model_path = MODEL_PATH if MODEL_PATH.exists() else "all-MiniLM-L6-v2"
    log.info("Loading model from: %s", model_path)
    return SentenceTransformer(str(model_path))


# ── FAISS index build (run once) ───────────────────────────────────────────────

def build_and_save_index(candidates_path: Path = CANDIDATES_JSONL) -> None:
    """
    Build FAISS index from full candidates.jsonl and save artifacts.
    Run this ONCE before the pipeline: python -m src.retrieval --build
    """
    import faiss

    log.info("Building FAISS index from %s", candidates_path)

    with open(candidates_path, "r", encoding="utf-8") as f:
        candidates = [json.loads(line) for line in f if line.strip()]

    log.info("Loaded %d candidates for indexing", len(candidates))

    texts = [build_retrieval_document(c) for c in candidates]
    model = _load_model()

    log.info("Encoding %d documents...", len(texts))
    embeddings = model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Validate
    assert embeddings.shape == (len(candidates), EXPECTED_DIM), \
        f"Shape mismatch: {embeddings.shape}"
    assert not np.isnan(embeddings).any(), "NaN in embeddings"

    # Build index
    index = faiss.IndexFlatIP(EXPECTED_DIM)
    index.add(embeddings)
    log.info("FAISS index built: %d vectors", index.ntotal)

    # Save artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(ARTIFACTS_DIR / "candidate_index.faiss"))

    candidate_mapping = {i: c["candidate_id"] for i, c in enumerate(candidates)}
    candidate_metadata = {c["candidate_id"]: c for c in candidates}

    with open(ARTIFACTS_DIR / "candidate_mapping.pkl", "wb") as f:
        pickle.dump(candidate_mapping, f)
    with open(ARTIFACTS_DIR / "candidate_metadata.pkl", "wb") as f:
        pickle.dump(candidate_metadata, f)

    log.info("Artifacts saved to %s", ARTIFACTS_DIR)


# ── FAISS search ───────────────────────────────────────────────────────────────

def _load_artifacts():
    import faiss
    index = faiss.read_index(str(ARTIFACTS_DIR / "candidate_index.faiss"))
    with open(ARTIFACTS_DIR / "candidate_mapping.pkl", "rb") as f:
        mapping = pickle.load(f)
    with open(ARTIFACTS_DIR / "candidate_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, mapping, metadata


def _search(jd_text: str, model, index, mapping, top_k: int = TOP_K_RETRIEVE) -> dict[str, float]:
    """Returns {candidate_id: similarity_score} for top_k results."""
    import faiss
    top_k = min(top_k, index.ntotal)
    qe    = model.encode([jd_text], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(qe, top_k)

    scores = {}
    for faiss_idx, dist in zip(indices[0], distances[0]):
        if faiss_idx >= 0:
            cid = mapping[faiss_idx]
            scores[cid] = float(dist)
    return scores


# ── Pipeline integration function ──────────────────────────────────────────────

def run_retrieval(df: pl.DataFrame) -> pl.DataFrame:
    """
    Pipeline integration point.

    Takes the filtered DataFrame from parser.py,
    scores each candidate via FAISS cosine similarity against the JD,
    returns the same DataFrame with 'semantic_score' column added.

    Candidates not in the FAISS index get score 0.0.
    """
    # Check artifacts exist
    index_path = ARTIFACTS_DIR / "candidate_index.faiss"
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}.\n"
            f"Run first: python -m src.retrieval --build"
        )

    log.info("Loading FAISS artifacts...")
    index, mapping, metadata = _load_artifacts()
    model = _load_model()
    jd    = load_job_description()

    log.info("Searching FAISS index (top %d)...", TOP_K_RETRIEVE)
    scores = _search(jd, model, index, mapping, top_k=TOP_K_RETRIEVE)

    # Normalise scores to [0, 1]  (cosine sim with normalized vecs is already [-1,1])
    if scores:
        min_s = min(scores.values())
        max_s = max(scores.values())
        rng   = max_s - min_s if max_s > min_s else 1.0
        scores = {cid: (s - min_s) / rng for cid, s in scores.items()}

    # Map scores onto the DataFrame
    candidate_ids   = df["candidate_id"].to_list()
    semantic_scores = [round(scores.get(cid, 0.0), 6) for cid in candidate_ids]

    df = df.with_columns(
        pl.Series("semantic_score", semantic_scores, dtype=pl.Float64)
    )

    matched = sum(1 for s in semantic_scores if s > 0)
    log.info(
        "Semantic scores assigned: %d matched / %d total (%.1f%%)",
        matched, len(candidate_ids), 100 * matched / max(len(candidate_ids), 1),
    )
    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [retrieval] %(message)s")

    if "--build" in sys.argv:
        path = SAMPLE_CANDIDATES if "--sample" in sys.argv else CANDIDATES_JSONL
        build_and_save_index(Path(path))
    else:
        from src.parser import run_parser
        use_sample = "--sample" in sys.argv
        df = run_parser(use_sample=use_sample)
        df = run_retrieval(df)
        print(f"\nTop 10 by semantic score:")
        print(df.sort("semantic_score", descending=True)
              .select(["candidate_id", "semantic_score", "years_of_experience"])
              .head(10))