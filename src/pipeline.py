"""
pipeline.py — Master orchestrator connecting Stages 1-4.

Usage:
    python3 -m src.pipeline                   # full run on candidates.jsonl
    python3 -m src.pipeline --sample          # quick run on sample_candidates.json
    python3 -m src.pipeline --mock            # skip retrieval.py (mock semantic scores)
    python3 -m src.pipeline --sample --mock   # dev mode: sample + mock

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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CANDIDATES_JSONL, SAMPLE_CANDIDATES,
    SUBMISSION_PATH, OUTPUT_DIR,
    REQUIRED_SKILLS, NLP_IR_SKILLS,
)

log = logging.getLogger(__name__)


# ── Stage runners ──────────────────────────────────────────────────────────────

def stage1_parse(data_path: Path) -> pl.DataFrame:
    from src.parser import run_parser
    log.info("=== STAGE 1: Parsing & filtering ===")
    t0 = time.time()
    df = run_parser(candidates_path=data_path)
    log.info("Stage 1 done in %.1fs — %d candidates survive", time.time() - t0, df.height)
    return df


def stage2_retrieve(df: pl.DataFrame) -> pl.DataFrame:
    log.info("=== STAGE 2: Semantic retrieval ===")
    t0 = time.time()
    try:
        from src.retrieval import run_retrieval
        df = run_retrieval(df)
        log.info("Stage 2 done in %.1fs (real semantic scores)", time.time() - t0)
    except (ImportError, FileNotFoundError) as e:
        log.warning("retrieval.py not ready (%s) — using mock semantic scores", e)
        df = _mock_semantic_scores(df)
        log.info("Stage 2 done in %.1fs (mock scores)", time.time() - t0)
    return df


def stage2_mock(df: pl.DataFrame) -> pl.DataFrame:
    log.info("=== STAGE 2: Mock semantic scores (--mock flag) ===")
    df = _mock_semantic_scores(df)
    log.info("Mock scores assigned to %d candidates", df.height)
    return df


def stage3_rank(df: pl.DataFrame) -> pl.DataFrame:
    from src.ranker import rank_candidates, build_submission
    log.info("=== STAGE 3: Behavioral ranking ===")
    t0 = time.time()
    ranked     = rank_candidates(df)
    submission = build_submission(ranked)
    log.info("Stage 3 done in %.1fs — top %d selected", time.time() - t0, submission.height)
    return submission


def stage4_generate(submission: pl.DataFrame) -> pl.DataFrame:
    log.info("=== STAGE 4: Reasoning generation ===")
    t0 = time.time()
    try:
        from src.generator import run_generator
        submission = run_generator(submission)
        log.info("Stage 4 done in %.1fs (LLM reasoning)", time.time() - t0)
    except (ImportError, Exception) as e:
        log.warning("generator.py not available (%s) — keeping heuristic reasoning", e)
    return submission


# ── Mock semantic scoring ──────────────────────────────────────────────────────

def _mock_semantic_scores(df: pl.DataFrame) -> pl.DataFrame:
    import json, random
    random.seed(42)

    rows = df.to_dicts()
    for r in rows:
        candidate   = json.loads(r.get("profile_json", "{}"))
        profile     = candidate.get("profile", {})
        skills      = candidate.get("skills", [])
        skill_names = {s.get("name", "").lower() for s in skills}
        relevant    = len(skill_names & (REQUIRED_SKILLS | NLP_IR_SKILLS))
        yoe         = float(profile.get("years_of_experience", 0))
        base        = 0.30 + min(relevant * 0.05, 0.40)
        yoe_bonus   = 0.10 if 5 <= yoe <= 9 else 0.0
        r["semantic_score"] = min(0.95, base + yoe_bonus + random.uniform(-0.05, 0.05))

    return pl.DataFrame(rows)


# ── Output ─────────────────────────────────────────────────────────────────────

def write_final(submission: pl.DataFrame, path: Path) -> None:
    from src.ranker import write_submission
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    write_submission(submission, path)


def print_summary(submission: pl.DataFrame, elapsed: float) -> None:
    rows = submission.to_dicts()
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Submission: {SUBMISSION_PATH}")
    print(f"{'='*60}\n")
    print("Top 10 candidates:\n")
    for r in rows[:10]:
        print(f"  #{r['rank']:>3}  {r['candidate_id']}  score={r['score']:.4f}")
        print(f"       {str(r['reasoning'])[:90]}...")
        print()


# ── Main ───────────────────────────────────────────────────────────────────────

def run_pipeline(
    data_path: Path,
    use_mock: bool = False,
    output_path: Path = SUBMISSION_PATH,
) -> pl.DataFrame:
    wall_start = time.time()

    df         = stage1_parse(data_path)
    df         = stage2_mock(df) if use_mock else stage2_retrieve(df)
    submission = stage3_rank(df)
    submission = stage4_generate(submission)
    write_final(submission, output_path)

    elapsed = time.time() - wall_start
    print_summary(submission, elapsed)
    log.info("Total wall time: %.1fs", elapsed)
    return submission


def main():
    ap = argparse.ArgumentParser(description="Redrob candidate ranking pipeline")
    ap.add_argument("--sample", action="store_true", help="Use sample_candidates.json")
    ap.add_argument("--mock",   action="store_true", help="Use mock semantic scores")
    ap.add_argument("--output", type=str, default=str(SUBMISSION_PATH))
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    data_path = SAMPLE_CANDIDATES if args.sample else CANDIDATES_JSONL

    if not Path(data_path).exists():
        log.error("Data file not found: %s", data_path)
        raise SystemExit(1)

    run_pipeline(
        data_path   = Path(data_path),
        use_mock    = args.mock,
        output_path = Path(args.output),
    )


if __name__ == "__main__":
    main()