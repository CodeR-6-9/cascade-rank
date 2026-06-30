"""
pipeline.py — Master Orchestrator for CascadeRank.
Executes Stages 1-4 within the 5-minute, 16GB RAM constraints.

Usage:
    python -m src.pipeline
"""

import time
import logging
import json
import gc
import csv
from pathlib import Path

import polars as pl
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Import our custom modules
from src.parser import run_parser
from src.ranker import run_ranker
from src.retrieval import load_job_description, EXPECTED_DIMENSIONS
from src.generator import ReasoningGenerator
from src.config import SUBMISSION_PATH, ROOT

JD_PATH = ROOT / "data" / "raw" / "job_description.docx"
MODEL_PATH = ROOT / "models" / "all-MiniLM-L6-v2"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [pipeline] %(message)s")
log = logging.getLogger(__name__)

def run_full_pipeline():
    start_time = time.time()
    log.info("Initiating Pipeline...")

    # ════════════════════════════════════════════════════════════════════════════════
    # STAGE 1: PARSER (Heuristics & Trap Filter)
    # ════════════════════════════════════════════════════════════════════════════════
    log.info("--- STAGE 1: Deterministic Filtering ---")
    s1_start = time.time()
    
    # run_parser natively loads from config.CANDIDATES_JSONL
    passed_df = run_parser() 
    candidates_dicts = passed_df.to_dicts()
    
    log.info(f"Stage 1 completed in {time.time() - s1_start:.2f}s. Surviving candidates: {len(candidates_dicts)}")

    # ════════════════════════════════════════════════════════════════════════════════
    # STAGE 2: RETRIEVAL (Semantic Search)
    # ════════════════════════════════════════════════════════════════════════════════
    log.info("--- STAGE 2: Dense Semantic Retrieval ---")
    s2_start = time.time()
    
    # 1. Load JD text
    jd_text = load_job_description(JD_PATH)
    
    # 2. Extract text for embeddings
    texts_to_embed = [c["text_to_embed"] for c in candidates_dicts]
    candidate_ids = [c["candidate_id"] for c in candidates_dicts]
    
    # 3. Initialize model and embed
    log.info("Loading SentenceTransformer...")
    embed_model = SentenceTransformer(str(MODEL_PATH))
    
    log.info("Generating candidate embeddings...")
    embeddings = embed_model.encode(
        texts_to_embed,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    
    # 4. Build FAISS Index in-memory
    index = faiss.IndexFlatIP(EXPECTED_DIMENSIONS)
    index.add(embeddings)
    
    # 5. Embed JD and search
    jd_embedding = embed_model.encode(
        [jd_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Get top 300 candidates to pass to Ranker
    top_k = min(300, index.ntotal)
    distances, indices = index.search(jd_embedding, top_k)
    
    # Merge semantic scores back into the candidate dictionaries
    semantic_results = []
    for i, faiss_idx in enumerate(indices[0]):
        if faiss_idx == -1: continue
        
        cid = candidate_ids[faiss_idx]
        # Find original parsed dictionary
        original_dict = next(item for item in candidates_dicts if item["candidate_id"] == cid)
        original_dict["semantic_score"] = float(distances[0][i])
        semantic_results.append(original_dict)

    log.info(f"Stage 2 completed in {time.time() - s2_start:.2f}s.")
    
    # RAM CLEARANCE: Delete heavy embedding objects before loading LLM
    del embed_model
    del index
    del embeddings
    gc.collect()
    log.info("Cleared embedding model from memory.")

    # ════════════════════════════════════════════════════════════════════════════════
    # STAGE 3: RANKER (Behavioral Multipliers)
    # ════════════════════════════════════════════════════════════════════════════════
    log.info("--- STAGE 3: Behavioral Re-Ranking ---")
    s3_start = time.time()
    
    # run_ranker handles the multipliers and returns the final Top 100 DataFrame
    final_100_df = run_ranker(semantic_results, output_path=SUBMISSION_PATH)
    
    log.info(f"Stage 3 completed in {time.time() - s3_start:.2f}s. Extracted Top {len(final_100_df)}.")

    # ════════════════════════════════════════════════════════════════════════════════
    # STAGE 4: GENERATOR (LLM Reasoning)
    # ════════════════════════════════════════════════════════════════════════════════
    log.info("--- STAGE 4: Local Reasoning Generation ---")
    s4_start = time.time()
    
    log.info("Booting Phi-3 GGUF model via llama.cpp...")
    generator = ReasoningGenerator(n_ctx=1024, n_threads=4) 
    
    # Convert the Top 100 DF to dicts to iterate through
    top_100_dicts = final_100_df.to_dicts()
    
    enriched_rows = []
    for idx, row in enumerate(top_100_dicts):
        # Rehydrate the raw candidate JSON so the generator has access to formatting helpers
        raw_candidate = json.loads(row["profile_json"])
        record = {
            "candidate_id": row["candidate_id"],
            "rank": row["rank"],
            "score": row["score"],
            "candidate": raw_candidate 
        }
        
        # Generate the reasoning
        result = generator.generate(record, jd_text)
        
        enriched_rows.append({
            "candidate_id": result["candidate_id"],
            "rank": result["rank"],
            "score": result["score"],
            "reasoning": result["reasoning"]
        })
        
        if (idx + 1) % 10 == 0:
            log.info(f"Generated reasoning for {idx + 1}/100 candidates...")

    log.info(f"Stage 4 completed in {time.time() - s4_start:.2f}s.")

    # ════════════════════════════════════════════════════════════════════════════════
    # FINAL EXPORT
    # ════════════════════════════════════════════════════════════════════════════════
    # Overwrite the CSV created in Stage 3 with the new LLM reasonings
    with open(SUBMISSION_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(enriched_rows)

    total_time = time.time() - start_time
    log.info(f"🎉 Pipeline Complete! Total Execution Time: {total_time:.2f}s")
    log.info(f"Final output saved to {SUBMISSION_PATH}")

if __name__ == "__main__":
    run_full_pipeline()