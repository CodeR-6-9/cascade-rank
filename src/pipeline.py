"""
pipeline.py — Master orchestrator connecting Stages 1-4.

Usage:
    python3 src/pipeline.py                          # full run on candidates.jsonl
    python3 src/pipeline.py --sample                 # quick run on sample_candidates.json
    python3 src/pipeline.py --mock                   # skip retrieval.py (mock semantic scores)
    python3 src/pipeline.py --sample --mock          # dev mode: sample + mock

Stages:
    1. parser.py     — load + hard filter 100K candidates
    2. retrieval.py  — sentence-transformers + FAISS semantic scoring  (Person B)
    3. ranker.py     — behavioral signal multipliers → final score
    4. generator.py  — LLM reasoning generation                        (Person B)
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import polars as pl

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CANDIDATES_JSONL, SAMPLE_CANDIDATES,
    SUBMISSION_PATH, OUTPUT_DIR,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def stage1_parse(data_path: str) -> pl.DataFrame:
    from parser import run_parser
    log.info("=== STAGE 1: Parsing & filtering ===")
    t0 = time.time()
    df = run_parser(data_path)
    log.info("Stage 1 done in %.1fs — %d candidates survive", time.time() - t0, df.height)
    return df


def stage2_retrieve(df: pl.DataFrame) -> pl.DataFrame:
    """
    Person B's retrieval.py — adds `semantic_score` column.
    Falls back to mock scores if retrieval.py isn't ready yet.
    """
    log.info("=== STAGE 2: Semantic retrieval ===")
    t0 = time.time()
    try:
        from retrieval import run_retrieval
        df = run_retrieval(df)
        log.info("Stage 2 done in %.1fs (real semantic scores)", time.time() - t0)
    except ImportError:
        log.warning("retrieval.py not available — using mock semantic scores")
        df = _mock_semantic_scores(df)
        log.info("Stage 2 done in %.1fs (mock scores)", time.time() - t0)
    return df


def stage2_mock(df: pl.DataFrame) -> pl.DataFrame:
    """Explicit mock path — bypasses retrieval.py entirely."""
    log.info("=== STAGE 2: Mock semantic scores (--mock flag) ===")
    df = _mock_semantic_scores(df)
    log.info("Mock scores assigned to %d candidates", df.height)
    return df


def stage3_rank(df: pl.DataFrame) -> pl.DataFrame:
    from ranker import rank_candidates, build_submission
    log.info("=== STAGE 3: Behavioral ranking ===")
    t0 = time.time()
    ranked     = rank_candidates(df)
    submission = build_submission(ranked)
    log.info("Stage 3 done in %.1fs — top 100 selected", time.time() - t0)
    return submission


def stage4_generate(submission: pl.DataFrame) -> pl.DataFrame:
    """
    Person B's generator.py — replaces mock reasoning with LLM-generated
    1-2 sentence justifications.
    Falls back to existing reasoning if generator.py isn't ready.
    """
    log.info("=== STAGE 4: Reasoning generation ===")
    t0 = time.time()
    try:
        from generator import run_generator
        submission = run_generator(submission)
        log.info("Stage 4 done in %.1fs (LLM reasoning)", time.time() - t0)
    except ImportError:
        log.warning("generator.py not available — keeping heuristic reasoning")
    return submission


# ---------------------------------------------------------------------------
# Mock semantic scoring (used when retrieval.py not ready)
# ---------------------------------------------------------------------------

def _mock_semantic_scores(df: pl.DataFrame) -> pl.DataFrame:
    """
    Simple heuristic stand-in for Person B's semantic scores.
    Counts relevant skill matches against JD required skills.
    """
    import json
    import random
    random.seed(42)

    from config import REQUIRED_SKILLS, NLP_IR_SKILLS

    rows = df.to_dicts()
    for r in rows:
        candidate   = json.loads(r.get("profile_json", "{}"))
        profile     = candidate.get("profile", {})
        skills      = candidate.get("skills", [])
        skill_names = {s.get("name", "").lower() for s in skills}

        relevant_count = len(skill_names & (REQUIRED_SKILLS | NLP_IR_SKILLS))
        yoe            = profile.get("years_of_experience", 0)

        base      = 0.30 + min(relevant_count * 0.05, 0.40)
        yoe_bonus = 0.10 if 5 <= yoe <= 9 else 0.0
        noise     = random.uniform(-0.05, 0.05)

        r["semantic_score"] = min(0.95, base + yoe_bonus + noise)

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_final(submission: pl.DataFrame, path: str = SUBMISSION_PATH) -> None:
    from ranker import write_submission
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    write_submission(submission, path)


def print_summary(submission: pl.DataFrame, elapsed: float) -> None:
    rows = submission.to_dicts()
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Submission: {SUBMISSION_PATH}")
    print(f"{'='*60}")
    print(f"\nTop 10 candidates:\n")
    for r in rows[:10]:
        print(f"  #{r['rank']:>3}  {r['candidate_id']}  score={r['score']:.4f}")
        print(f"       {r['reasoning'][:90]}...")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pipeline(
    data_path: str,
    use_mock: bool = False,
    output_path: str = SUBMISSION_PATH,
) -> pl.DataFrame:
    """
    Full 4-stage pipeline.

    Args:
        data_path:   Path to candidates.jsonl or sample_candidates.json
        use_mock:    If True, skip retrieval.py and use mock semantic scores
        output_path: Where to write the final CSV

    Returns:
        submission DataFrame (100 rows)
    """
    wall_start = time.time()

    # Stage 1 — parse + filter
    df = stage1_parse(data_path)

    # Stage 2 — semantic scoring
    if use_mock:
        df = stage2_mock(df)
    else:
        df = stage2_retrieve(df)   # falls back to mock if retrieval.py missing

    # Stage 3 — behavioral ranking → top 100
    submission = stage3_rank(df)

    # Stage 4 — LLM reasoning (optional, falls back gracefully)
    submission = stage4_generate(submission)

    # Write output
    write_final(submission, output_path)

    elapsed = time.time() - wall_start
    print_summary(submission, elapsed)

    log.info("Total wall time: %.1fs", elapsed)
    return submission


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Redrob candidate ranking pipeline"
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Run on sample_candidates.json instead of full candidates.jsonl"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock semantic scores (skip retrieval.py)"
    )
    parser.add_argument(
        "--output", type=str, default=SUBMISSION_PATH,
        help=f"Output CSV path (default: {SUBMISSION_PATH})"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    data_path = SAMPLE_CANDIDATES if args.sample else CANDIDATES_JSONL

    if not Path(data_path).exists():
        log.error("Data file not found: %s", data_path)
        log.error("Run: cp <bundle>/candidates.jsonl data/raw/")
        raise SystemExit(1)

    run_pipeline(
        data_path  = data_path,
        use_mock   = args.mock,
        output_path = args.output,
    )


if __name__ == "__main__":
    main()