"""
ranker.py — Stage 3: Apply behavioral signal multipliers to semantic scores.

Input  (from retrieval.py via pipeline.py):
  Polars DataFrame with columns from PARSER_OUTPUT_COLUMNS + semantic_score

Output:
  SUBMISSION_PATH / final_submission.csv  (validated format)

Run standalone (mock semantic scores):
  python -m src.ranker --sample
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Union

import polars as pl

import os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    TODAY, TOP_N, SUBMISSION_PATH, SUBMISSION_COLS,
    ALLOWED_LOCATIONS, PREFERRED_LOCATIONS,
    GHOST_INACTIVITY_DAYS, STRONG_RECENCY_DAYS,
    IDEAL_NOTICE_DAYS, LONG_NOTICE_DAYS, HARD_NOTICE_CUTOFF,
    MIN_RESPONSE_RATE, GOOD_RESPONSE_RATE, EXCELLENT_RESPONSE_RATE,
    GOOD_GITHUB_SCORE, STRONG_GITHUB_SCORE,
    MIN_PROFILE_COMPLETENESS, GOOD_PROFILE_COMPLETENESS,
    SALARY_MIN_REALISTIC, SALARY_MAX_REALISTIC,
    AVAILABILITY_MAX, AVAILABILITY_MIN,
    ENGAGEMENT_MAX, ENGAGEMENT_MIN,
    FIT_MAX, FIT_MIN,
    BOOST_OPEN_TO_WORK, BOOST_RECENT_ACTIVE_30D,
    BOOST_GITHUB_GOOD, BOOST_GITHUB_STRONG,
    BOOST_SAVED_BY_RECRUITERS, BOOST_STRONG_RESPONSE_RATE,
    BOOST_HIGH_INTERVIEW_RATE, BOOST_PREFERRED_LOCATION,
    BOOST_SHORT_NOTICE, BOOST_VERIFIED_CONTACT,
    PENALTY_LONG_NOTICE_60, PENALTY_LONG_NOTICE_90,
    PENALTY_GHOST_INACTIVE, PENALTY_LOW_RESPONSE,
    PENALTY_LOW_PROFILE, PENALTY_SALARY_MISMATCH,
    PENALTY_HYPE_SKILLS_ONLY,
    HYPE_ONLY_SKILLS, REQUIRED_SKILLS, NLP_IR_SKILLS,
    JD_TARGET_YOE_MIN, JD_TARGET_YOE_MAX,
    PURE_SERVICES_FIRMS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ranker] %(message)s")
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def _days_inactive(signals: dict) -> int:
    try:
        last = date.fromisoformat(signals["last_active_date"])
        return max(0, (TODAY - last).days)
    except (KeyError, ValueError):
        return 999


# ════════════════════════════════════════════════════════════════════════════════
# HARD DISQUALIFIERS  (return True = disqualify → score forced to 0)
# Run before any multiplier computation. Fast-fail.
# ════════════════════════════════════════════════════════════════════════════════

def _hard_ghost(signals: dict) -> bool:
    """
    Hard ghost: inactive > 6 months AND not open-to-work AND < 5% response rate.
    AND logic avoids punishing someone who just forgot to log in.
    """
    days  = _days_inactive(signals)
    open_ = signals.get("open_to_work_flag", False)
    rr    = float(signals.get("recruiter_response_rate", 0.0))
    return days > GHOST_INACTIVITY_DAYS and not open_ and rr < 0.05


def _hard_location(profile: dict, signals: dict) -> bool:
    """Outside India AND won't relocate → disqualify."""
    country = profile.get("country", "").lower().strip()
    willing = signals.get("willing_to_relocate", False)
    if country in ("india", "in", ""):
        return False
    return not willing


# ════════════════════════════════════════════════════════════════════════════════
# MULTIPLIER 1: AVAILABILITY
# How reachable is this candidate right now?
# Range: [AVAILABILITY_MIN, AVAILABILITY_MAX]
# ════════════════════════════════════════════════════════════════════════════════

def availability_multiplier(signals: dict) -> float:
    mult  = 1.0
    days  = _days_inactive(signals)

    # Recency
    if days <= STRONG_RECENCY_DAYS:
        mult += BOOST_RECENT_ACTIVE_30D
    elif days > GHOST_INACTIVITY_DAYS:
        mult += PENALTY_GHOST_INACTIVE           # negative constant
    else:
        # Smooth linear decay between 30d and 180d
        t     = (days - STRONG_RECENCY_DAYS) / (GHOST_INACTIVITY_DAYS - STRONG_RECENCY_DAYS)
        mult -= t * 0.20

    # Open-to-work flag
    if signals.get("open_to_work_flag", False):
        mult += BOOST_OPEN_TO_WORK

    # Notice period (4 bands, not binary)
    notice = int(signals.get("notice_period_days", 30))
    if notice <= IDEAL_NOTICE_DAYS:
        mult += BOOST_SHORT_NOTICE
    elif notice > HARD_NOTICE_CUTOFF:
        mult += PENALTY_LONG_NOTICE_90           # > 90d
    elif notice > LONG_NOTICE_DAYS:
        mult += PENALTY_LONG_NOTICE_60           # 61–90d
    # 31–60d: no change

    # Response rate
    rr = float(signals.get("recruiter_response_rate", 0.5))
    if rr < MIN_RESPONSE_RATE:
        mult += PENALTY_LOW_RESPONSE
    elif rr >= EXCELLENT_RESPONSE_RATE:
        mult += BOOST_STRONG_RESPONSE_RATE

    # Verified contact (reachability proxy)
    if signals.get("verified_email", False) and signals.get("verified_phone", False):
        mult += BOOST_VERIFIED_CONTACT

    return float(max(AVAILABILITY_MIN, min(AVAILABILITY_MAX, mult)))


# ════════════════════════════════════════════════════════════════════════════════
# MULTIPLIER 2: ENGAGEMENT
# Platform engagement quality — are recruiters interested? Is profile credible?
# Range: [ENGAGEMENT_MIN, ENGAGEMENT_MAX]
# ════════════════════════════════════════════════════════════════════════════════

def engagement_multiplier(signals: dict) -> float:
    mult = 1.0

    # GitHub — wide range (0.12 to 0.20 boost) because it's a real differentiator
    gh = float(signals.get("github_activity_score", -1))
    if gh >= STRONG_GITHUB_SCORE:
        mult += BOOST_GITHUB_STRONG
    elif gh >= GOOD_GITHUB_SCORE:
        mult += BOOST_GITHUB_GOOD
    # -1 (not linked) = neutral. Not boosted but not penalised here
    # (soft penalty is implicit from missing boost)

    # Recruiters saving profile = third-party signal of desirability
    saved = int(signals.get("saved_by_recruiters_30d", 0))
    if saved >= 3:
        mult += BOOST_SAVED_BY_RECRUITERS

    # Interview seriousness
    icr = float(signals.get("interview_completion_rate", 0.5))
    if icr >= 0.80:
        mult += BOOST_HIGH_INTERVIEW_RATE

    # Profile completeness
    pcs = float(signals.get("profile_completeness_score", 100))
    if pcs < MIN_PROFILE_COMPLETENESS:
        mult += PENALTY_LOW_PROFILE

    # Skill assessment scores on Redrob platform
    assessments = signals.get("skill_assessment_scores", {})
    relevant = [
        v for k, v in assessments.items()
        if k.lower() in (REQUIRED_SKILLS | NLP_IR_SKILLS)
    ]
    if relevant:
        avg = sum(relevant) / len(relevant)
        if avg >= 75:
            mult += 0.10
        elif avg >= 60:
            mult += 0.05

    return float(max(ENGAGEMENT_MIN, min(ENGAGEMENT_MAX, mult)))


# ════════════════════════════════════════════════════════════════════════════════
# MULTIPLIER 3: JD FIT
# Location alignment, salary, skill quality (not keyword count), hype-penalty
# Range: [FIT_MIN, FIT_MAX]
# ════════════════════════════════════════════════════════════════════════════════

def fit_multiplier(profile: dict, signals: dict, skills: list[dict]) -> float:
    mult = 1.0

    # Location preference
    location = profile.get("location", "").lower()
    country  = profile.get("country", "").lower()
    willing  = signals.get("willing_to_relocate", False)

    if any(pref in location for pref in PREFERRED_LOCATIONS):
        mult += BOOST_PREFERRED_LOCATION
    elif country in ("india", "in") and willing:
        mult += BOOST_PREFERRED_LOCATION * 0.5   # partial for relocation within India

    # Salary alignment
    sal     = signals.get("expected_salary_range_inr_lpa", {})
    sal_max = float(sal.get("max", 30))
    if sal_max > SALARY_MAX_REALISTIC or sal_max < SALARY_MIN_REALISTIC:
        mult += PENALTY_SALARY_MISMATCH

    # Hype-skills-only: LangChain/GPT-wrapper profile with no real retrieval skills
    skill_names  = {s.get("name", "").lower() for s in skills}
    has_required = bool(skill_names & REQUIRED_SKILLS)
    has_hype     = bool(skill_names & HYPE_ONLY_SKILLS)
    if has_hype and not has_required:
        mult += PENALTY_HYPE_SKILLS_ONLY

    # YOE sweet-spot bonus (5–9 yrs from JD)
    yoe = float(profile.get("years_of_experience", 0))
    if JD_TARGET_YOE_MIN <= yoe <= JD_TARGET_YOE_MAX:
        mult += 0.05

    return float(max(FIT_MIN, min(FIT_MAX, mult)))


# ════════════════════════════════════════════════════════════════════════════════
# HONEYPOT PENALTY
# ════════════════════════════════════════════════════════════════════════════════

def honeypot_penalty(hp_score: float) -> float:
    """Convert 0–1 suspicion score → multiplier. 0 = clean (1.0), 1 = honeypot (0.20)."""
    return 1.0 - (hp_score * 0.80)


# ════════════════════════════════════════════════════════════════════════════════
# FINAL SCORE
# ════════════════════════════════════════════════════════════════════════════════

def compute_final_score(row: dict) -> float:
    """
    final_score = semantic × availability × engagement × fit × honeypot_penalty
    Hard disqualifiers → 0.0 immediately.
    """
    semantic = float(row.get("semantic_score", 0.0))
    hp_score = float(row.get("honeypot_score", 0.0))

    signals  = json.loads(row.get("signals_json", "{}"))
    full_c   = json.loads(row.get("profile_json", "{}"))
    profile  = full_c.get("profile", {})
    skills   = full_c.get("skills", [])

    # Hard disqualifiers — score = 0 (sink to bottom, won't appear in top 100)
    if _hard_ghost(signals):
        return 0.0
    if _hard_location(profile, signals):
        return 0.0

    avail  = availability_multiplier(signals)
    engage = engagement_multiplier(signals)
    fit    = fit_multiplier(profile, signals, skills)
    hp_pen = honeypot_penalty(hp_score)

    final = semantic * avail * engage * fit * hp_pen
    return round(float(max(0.0, min(1.0, final))), 6)


# ════════════════════════════════════════════════════════════════════════════════
# REASONING GENERATION  (template; Stage 4 / generator.py enriches this)
# ════════════════════════════════════════════════════════════════════════════════

def generate_reasoning(row: dict, rank: int) -> str:
    """
    Tier-based 1–2 sentence justification grounded in actual candidate facts.
    Stage 4 (generator.py) replaces this with LLM output for the final top 100.
    """
    full_c  = json.loads(row.get("profile_json", "{}"))
    profile = full_c.get("profile", {})
    signals = json.loads(row.get("signals_json", "{}"))
    skills  = full_c.get("skills", [])
    career  = full_c.get("career_history", [])

    title   = profile.get("current_title", "Engineer")
    company = profile.get("current_company", "")
    yoe     = float(profile.get("years_of_experience", 0))
    notice  = int(signals.get("notice_period_days", 30))
    days    = _days_inactive(signals)
    open_w  = signals.get("open_to_work_flag", False)
    github  = float(signals.get("github_activity_score", -1))
    rr      = float(signals.get("recruiter_response_rate", 0))

    # Relevant skills
    relevant = [
        s["name"] for s in skills
        if s.get("name", "").lower() in (REQUIRED_SKILLS | NLP_IR_SKILLS)
    ][:3]
    skill_str = ", ".join(relevant) if relevant else "applied ML"

    # Product company experience
    product_exp = [
        r["company"] for r in career
        if not any(f in r.get("company", "").lower() for f in PURE_SERVICES_FIRMS)
    ]

    company_ctx = f"at {company}" if company else ""
    s1 = (
        f"{yoe:.0f}-yr {title} {company_ctx}; skills: {skill_str}."
    )

    positives = []
    if open_w:                            positives.append("open to work")
    if notice <= 30:                      positives.append(f"short notice ({notice}d)")
    if github >= GOOD_GITHUB_SCORE:       positives.append(f"GitHub {github:.0f}/100")
    if product_exp:                       positives.append(f"product co exp")
    if rr >= EXCELLENT_RESPONSE_RATE:     positives.append(f"high response rate ({rr:.0%})")

    concerns = []
    if days > 90:                         concerns.append(f"inactive {days}d")
    if notice > 60:                       concerns.append(f"notice {notice}d")
    if not open_w:                        concerns.append("passive candidate")
    if github < 0:                        concerns.append("no GitHub")

    if rank <= 20:
        pos = "; ".join(positives[:2]) or "solid signals"
        con = f" Note: {concerns[0]}." if concerns else ""
        s2  = f"Strong fit — {pos}.{con}"
    elif rank <= 60:
        pos = positives[0] if positives else "some signals"
        con = f"; concern: {', '.join(concerns[:2])}" if concerns else ""
        s2  = f"Moderate fit — {pos}{con}."
    else:
        con = f"Gaps: {', '.join(concerns[:2])}" if concerns else "limited signals"
        s2  = f"Marginal — {con}; retained on skill adjacency."

    return f"{s1} {s2}"


# ════════════════════════════════════════════════════════════════════════════════
# MAIN RANKING PIPELINE
# ════════════════════════════════════════════════════════════════════════════════

def rank_candidates(df: pl.DataFrame) -> pl.DataFrame:
    """Score all candidates and return top TOP_N sorted by score desc."""
    rows   = df.to_dicts()
    scored = [{**r, "final_score": compute_final_score(r)} for r in rows]

    disq = sum(1 for r in scored if r["final_score"] == 0.0)
    log.info("Hard disqualified in Stage 3: %d", disq)

    scored_df = pl.DataFrame(scored)
    return (
        scored_df
        .sort(["final_score", "candidate_id"], descending=[True, False])
        .head(TOP_N)
    )


def build_submission(ranked: pl.DataFrame) -> pl.DataFrame:
    """Add rank (1–100), score, reasoning columns."""
    rows = ranked.to_dicts()
    out  = []
    for i, r in enumerate(rows):
        rank = i + 1
        out.append({
            "candidate_id": r["candidate_id"],
            "rank":         rank,
            "score":        round(r["final_score"], 4),
            "reasoning":    generate_reasoning(r, rank),
        })
    return pl.DataFrame(out)


def write_submission(submission: pl.DataFrame, path: Path = SUBMISSION_PATH) -> None:
    """Write CSV. Validates score monotonicity before writing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows      = submission.to_dicts()
    prev      = float("inf")
    for r in rows:
        if r["score"] > prev:
            log.warning("Non-monotonic scores at rank %d", r["rank"])
        prev = r["score"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    log.info("Submission written → %s (%d rows)", path, len(rows))


def run_ranker(
    df: Union[pl.DataFrame, list],
    output_path: Path | str = SUBMISSION_PATH,
) -> pl.DataFrame:
    """
    Full Stage 3 pipeline.
    Accepts: Polars DataFrame (from pipeline.py) OR list-of-dicts (from Person B).
    Returns and writes final submission DataFrame.
    """
    if isinstance(df, list):
        df = pl.DataFrame(df)

    if "semantic_score" not in df.columns:
        raise ValueError("'semantic_score' column missing — run retrieval.py first.")

    log.info("Stage 3: ranking %d candidates...", df.height)
    ranked     = rank_candidates(df)
    submission = build_submission(ranked)
    write_submission(submission, output_path)
    return submission


# ════════════════════════════════════════════════════════════════════════════════
# MOCK MODE — test Stage 3 without Person B's semantic scores
# ════════════════════════════════════════════════════════════════════════════════

def run_ranker_mock(df: pl.DataFrame, output_path: Path | str = SUBMISSION_PATH) -> pl.DataFrame:
    """Injects a heuristic mock_semantic_score so Stage 3 can be tested solo."""
    import random
    random.seed(42)

    def _mock(row: dict) -> float:
        full_c = json.loads(row.get("profile_json", "{}"))
        skills = full_c.get("skills", [])
        profile = full_c.get("profile", {})
        names   = {s.get("name", "").lower() for s in skills}
        rel     = len(names & (REQUIRED_SKILLS | NLP_IR_SKILLS))
        yoe     = float(profile.get("years_of_experience", 0))
        base    = 0.30 + min(rel * 0.05, 0.40)
        yoe_b   = 0.10 if 5 <= yoe <= 9 else 0.0
        return min(0.95, base + yoe_b + random.uniform(-0.05, 0.05))

    rows     = df.to_dicts()
    enriched = [{**r, "semantic_score": _mock(r)} for r in rows]
    return run_ranker(pl.DataFrame(enriched), output_path)


# ════════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from src.parser import run_parser

    use_sample = "--sample" in sys.argv
    df = run_parser(use_sample=use_sample)

    print(f"\nCandidates entering Stage 3: {df.height}")
    submission = run_ranker_mock(df)
    print(f"\nTop 10:")
    print(submission.head(10))
    print(f"\nScore range: {submission['score'].max():.4f} → {submission['score'].min():.4f}")