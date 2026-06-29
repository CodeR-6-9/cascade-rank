from __future__ import annotations
import argparse, logging, time, sys, os, json, random
from pathlib import Path
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CANDIDATES_JSONL, SAMPLE_CANDIDATES,
    SUBMISSION_PATH, OUTPUT_DIR,
    REQUIRED_SKILLS, NLP_IR_SKILLS,
)

log = logging.getLogger(__name__)

def stage1_parse(data_path):
    from src.parser import run_parser
    log.info("=== STAGE 1: Parsing & filtering ===")
    t0 = time.time()
    df = run_parser(candidates_path=Path(data_path))
    log.info("Stage 1 done in %.1fs — %d candidates survive", time.time()-t0, df.height)
    return df

def _mock_semantic_scores(df):
    random.seed(42)
    rows = df.to_dicts()
    for r in rows:
        c = json.loads(r.get("profile_json", "{}"))
        skills = {s.get("name","").lower() for s in c.get("skills",[])}
        yoe = float(c.get("profile",{}).get("years_of_experience", 0))
        rel = len(skills & (REQUIRED_SKILLS | NLP_IR_SKILLS))
        base = 0.30 + min(rel * 0.05, 0.40)
        r["semantic_score"] = min(0.95, base + (0.10 if 5<=yoe<=9 else 0) + random.uniform(-0.05,0.05))
    return pl.DataFrame(rows)

def stage2_retrieve(df, use_mock=False):
    log.info("=== STAGE 2: Semantic retrieval ===")
    if use_mock:
        log.info("Using mock semantic scores")
        return _mock_semantic_scores(df)
    try:
        from src.retrieval import run_retrieval
        return run_retrieval(df)
    except Exception as e:
        log.warning("retrieval.py not ready (%s) — using mock", e)
        return _mock_semantic_scores(df)

def stage3_rank(df):
    from src.ranker import rank_candidates, build_submission
    log.info("=== STAGE 3: Behavioral ranking ===")
    t0 = time.time()
    submission = build_submission(rank_candidates(df))
    log.info("Stage 3 done in %.1fs", time.time()-t0)
    return submission

def stage4_generate(submission):
    log.info("=== STAGE 4: Reasoning generation ===")
    try:
        from src.generator import run_generator
        return run_generator(submission)
    except Exception as e:
        log.warning("generator.py not available (%s) — keeping heuristic reasoning", e)
    return submission

def run_pipeline(data_path, use_mock=False, output_path=None):
    output_path = output_path or SUBMISSION_PATH
    wall = time.time()
    df = stage1_parse(data_path)
    df = stage2_retrieve(df, use_mock=use_mock)
    submission = stage3_rank(df)
    submission = stage4_generate(submission)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    from src.ranker import write_submission
    write_submission(submission, output_path)
    elapsed = time.time() - wall
    rows = submission.to_dicts()
    print(f"\n{'='*60}\n  Pipeline complete in {elapsed:.1f}s\n{'='*60}\n")
    for r in rows[:10]:
        print(f"  #{r['rank']:>3}  {r['candidate_id']}  score={r['score']:.4f}")
        print(f"       {str(r['reasoning'])[:90]}...\n")
    return submission

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--mock",   action="store_true")
    ap.add_argument("--output", type=str, default=str(SUBMISSION_PATH))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    data_path = SAMPLE_CANDIDATES if args.sample else CANDIDATES_JSONL
    if not Path(data_path).exists():
        log.error("Data file not found: %s", data_path)
        raise SystemExit(1)
    run_pipeline(data_path=data_path, use_mock=args.mock, output_path=Path(args.output))

if __name__ == "__main__":
    main()
