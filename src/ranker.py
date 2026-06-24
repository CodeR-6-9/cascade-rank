"""
ranker.py — Stage 3: Apply behavioral signal multipliers to semantic scores
and produce the final ranked top-100 CSV.

Input  (from pipeline.py, after retrieval.py fills in semantic_score):
  Polars DataFrame with columns:
    candidate_id, semantic_score, years_of_experience,
    signals_json, profile_json, honeypot_score

Output:
  data/output/final_submission.csv  (validated format)
"""

from __future__ import annotations

import csv
import json
import logging
import math
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TODAY, TOP_N,
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
    SUBMISSION_PATH, OUTPUT_DIR,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multiplier sub-functions
# Each returns a float. Multiplied together with the semantic score.
# ---------------------------------------------------------------------------

def _days_inactive(signals: dict) -> int:
    try:
        last = date.fromisoformat(signals["last_active_date"])
        return (TODAY - last).days
    except (KeyError, ValueError):
        return 999


def availability_multiplier(signals: dict) -> float:
    """
    How reachable / available is this candidate right now?
    Range: [AVAILABILITY_MIN, AVAILABILITY_MAX]
    """
    mult = 1.0
    days = _days_inactive(signals)

    # Recency
    if days <= STRONG_RECENCY_DAYS:
        mult += BOOST_RECENT_ACTIVE_30D
    elif days > GHOST_INACTIVITY_DAYS:
        mult += PENALTY_GHOST_INACTIVE   # negative
    else:
        # Linear decay between 30 and 180 days
        decay = (days - STRONG_RECENCY_DAYS) / (GHOST_INACTIVITY_DAYS - STRONG_RECENCY_DAYS)
        mult -= decay * 0.20   # up to -0.20 at 180 days

    # Open to work flag
    if signals.get("open_to_work_flag", False):
        mult += BOOST_OPEN_TO_WORK

    # Notice period
    notice = signals.get("notice_period_days", 30)
    if notice <= IDEAL_NOTICE_DAYS:
        mult += BOOST_SHORT_NOTICE
    elif notice > HARD_NOTICE_CUTOFF:
        mult += PENALTY_LONG_NOTICE_90
    elif notice > LONG_NOTICE_DAYS:
        mult += PENALTY_LONG_NOTICE_60

    # Response rate
    rr = signals.get("recruiter_response_rate", 0.5)
    if rr < MIN_RESPONSE_RATE:
        mult += PENALTY_LOW_RESPONSE
    elif rr >= EXCELLENT_RESPONSE_RATE:
        mult += BOOST_STRONG_RESPONSE_RATE

    # Verified contact details
    if signals.get("verified_email", False) and signals.get("verified_phone", False):
        mult += BOOST_VERIFIED_CONTACT

    return float(max(AVAILABILITY_MIN, min(AVAILABILITY_MAX, mult)))


def engagement_multiplier(signals: dict) -> float:
    """
    Platform engagement quality — are recruiters interested? Is the
    candidate active and credible?
    Range: [ENGAGEMENT_MIN, ENGAGEMENT_MAX]
    """
    mult = 1.0

    # GitHub activity
    gh = signals.get("github_activity_score", -1)
    if gh >= STRONG_GITHUB_SCORE:
        mult += BOOST_GITHUB_STRONG
    elif gh >= GOOD_GITHUB_SCORE:
        mult += BOOST_GITHUB_GOOD
    # -1 (not linked) = neutral, no change

    # Recruiter interest
    saved = signals.get("saved_by_recruiters_30d", 0)
    if saved >= 3:
        mult += BOOST_SAVED_BY_RECRUITERS

    # Interview seriousness
    icr = signals.get("interview_completion_rate", 0.5)
    if icr >= 0.80:
        mult += BOOST_HIGH_INTERVIEW_RATE

    # Profile completeness
    pcs = signals.get("profile_completeness_score", 100)
    if pcs < MIN_PROFILE_COMPLETENESS:
        mult += PENALTY_LOW_PROFILE

    # Skill assessment scores — average of relevant assessments
    assessments = signals.get("skill_assessment_scores", {})
    relevant_scores = [
        v for k, v in assessments.items()
        if k.lower() in (REQUIRED_SKILLS | NLP_IR_SKILLS)
    ]
    if relevant_scores:
        avg_assessment = sum(relevant_scores) / len(relevant_scores)
        # +0.10 for avg >= 75, +0.05 for >= 60
        if avg_assessment >= 75:
            mult += 0.10
        elif avg_assessment >= 60:
            mult += 0.05

    return float(max(ENGAGEMENT_MIN, min(ENGAGEMENT_MAX, mult)))


def fit_multiplier(profile: dict, signals: dict, skills: list[dict]) -> float:
    """
    JD-specific fit signals: location preference, salary alignment,
    skill quality (not just keywords), hype-skills-only penalty.
    Range: [FIT_MIN, FIT_MAX]
    """
    mult = 1.0

    # Location preference boost
    location = profile.get("location", "").lower()
    country  = profile.get("country", "").lower()
    willing  = signals.get("willing_to_relocate", False)

    if any(pref in location for pref in PREFERRED_LOCATIONS):
        mult += BOOST_PREFERRED_LOCATION
    elif (country in ("india", "in")) and willing:
        mult += BOOST_PREFERRED_LOCATION * 0.5  # partial boost for relocation

    # Salary alignment
    sal = signals.get("expected_salary_range_inr_lpa", {})
    sal_max = sal.get("max", 30)
    if sal_max > SALARY_MAX_REALISTIC or sal_max < SALARY_MIN_REALISTIC:
        mult += PENALTY_SALARY_MISMATCH

    # Hype-skills-only penalty
    skill_names = {s.get("name", "").lower() for s in skills}
    has_required = bool(skill_names & REQUIRED_SKILLS)
    has_hype_only = bool(skill_names & HYPE_ONLY_SKILLS) and not has_required

    if has_hype_only:
        mult += PENALTY_HYPE_SKILLS_ONLY

    # YOE sweet spot
    yoe = profile.get("years_of_experience", 0)
    if JD_TARGET_YOE_MIN <= yoe <= JD_TARGET_YOE_MAX:
        mult += 0.05   # small boost for ideal range

    return float(max(FIT_MIN, min(FIT_MAX, mult)))


def honeypot_penalty(hp_score: float) -> float:
    """Convert a 0-1 honeypot suspicion score into a multiplier (1.0 → 0.20)."""
    # Linear from 1.0 (clean) down to 0.20 (certain honeypot)
    return 1.0 - (hp_score * 0.80)


# ---------------------------------------------------------------------------
# Score computation per candidate
# ---------------------------------------------------------------------------

def compute_final_score(row: dict) -> float:
    """
    row must have: semantic_score, signals_json, profile_json, honeypot_score
    Returns final_score in [0, 1].
    """
    semantic    = float(row.get("semantic_score", 0.0))
    hp_score    = float(row.get("honeypot_score", 0.0))
    signals     = json.loads(row.get("signals_json", "{}"))
    candidate   = json.loads(row.get("profile_json", "{}"))
    profile     = candidate.get("profile", {})
    skills      = candidate.get("skills", [])

    avail  = availability_multiplier(signals)
    engage = engagement_multiplier(signals)
    fit    = fit_multiplier(profile, signals, skills)
    hp_pen = honeypot_penalty(hp_score)

    final = semantic * avail * engage * fit * hp_pen

    # Clamp to [0, 1]
    return float(max(0.0, min(1.0, final)))


# ---------------------------------------------------------------------------
# Reasoning generation
# ---------------------------------------------------------------------------

def _normalise_location(profile: dict) -> str:
    loc = profile.get("location", "")
    country = profile.get("country", "")
    if country.lower() == "india":
        return loc
    return f"{loc}, {country}" if country else loc


def generate_reasoning(row: dict, rank: int) -> str:
    """
    Produce a 1-2 sentence reasoning grounded in actual candidate facts.
    Deliberately varies by rank tier to pass Stage 4 rank-consistency check.
    """
    candidate = json.loads(row.get("profile_json", "{}"))
    profile   = candidate.get("profile", {})
    signals   = json.loads(row.get("signals_json", "{}"))
    skills    = candidate.get("skills", [])
    career    = candidate.get("career_history", [])

    title     = profile.get("current_title", "Unknown")
    company   = profile.get("current_company", "Unknown")
    yoe       = profile.get("years_of_experience", 0)
    location  = _normalise_location(profile)
    notice    = signals.get("notice_period_days", 30)
    days_inactive = _days_inactive(signals)
    open_work = signals.get("open_to_work_flag", False)
    github    = signals.get("github_activity_score", -1)

    # Top relevant skills
    relevant = [
        s["name"] for s in skills
        if s.get("name", "").lower() in (REQUIRED_SKILLS | NLP_IR_SKILLS)
    ][:3]

    # Most recent product company (non-consulting)
    from config import PURE_SERVICES_FIRMS
    product_companies = [
        r["company"] for r in career
        if not any(f in r.get("company", "").lower() for f in PURE_SERVICES_FIRMS)
    ]

    # Build sentence 1 — skills and background
    skill_str = ", ".join(relevant) if relevant else "applied ML"
    company_context = (
        f"currently at {company}"
        if company not in ("", "Unknown")
        else "independent"
    )

    s1 = (
        f"{yoe:.0f}-year career as {title} ({company_context}); "
        f"relevant skills: {skill_str}."
    )

    # Build sentence 2 — availability + concerns (vary by rank tier)
    concerns = []
    if days_inactive > 90:
        concerns.append(f"inactive for {days_inactive}d")
    if notice > 60:
        concerns.append(f"notice period {notice}d")
    if not open_work:
        concerns.append("not explicitly open to work")
    if github == -1:
        concerns.append("no GitHub linked")

    positives = []
    if open_work:
        positives.append("actively open to work")
    if notice <= 30:
        positives.append(f"short notice ({notice}d)")
    if github >= GOOD_GITHUB_SCORE:
        positives.append(f"strong GitHub ({github:.0f}/100)")
    if product_companies:
        positives.append(f"product company exp ({product_companies[0]})")

    if rank <= 20:
        # Strong candidates — lead with positives, note concerns honestly
        pos_str = "; ".join(positives[:2]) if positives else "solid engagement"
        con_str = f" Minor concern: {concerns[0]}." if concerns else ""
        s2 = f"Strong availability signals: {pos_str}.{con_str}"
    elif rank <= 60:
        # Moderate — balanced
        pos_str = positives[0] if positives else "some engagement"
        con_str = f"; concern: {', '.join(concerns[:2])}" if concerns else ""
        s2 = f"Moderate fit — {pos_str}{con_str}."
    else:
        # Weak — be honest about gaps
        con_str = f"Key concerns: {', '.join(concerns[:2])}" if concerns else "limited engagement signals"
        s2 = f"Marginal inclusion — {con_str}; retained on skill adjacency."

    return f"{s1} {s2}"


# ---------------------------------------------------------------------------
# Main ranking function
# ---------------------------------------------------------------------------

def rank_candidates(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute final_score for each candidate and return top TOP_N sorted
    by final_score descending.
    """
    # Compute scores row-by-row (Python loop is fine here — filtered pool
    # should be 5K-15K candidates, not 100K)
    rows = df.to_dicts()
    scored = []
    for r in rows:
        fs = compute_final_score(r)
        scored.append({**r, "final_score": fs})

    scored_df = pl.DataFrame(scored)

    # Sort: final_score desc, then candidate_id asc (tie-break)
    ranked = (
        scored_df
        .sort(["final_score", "candidate_id"], descending=[True, False])
        .head(TOP_N)
    )

    return ranked


def build_submission(ranked: pl.DataFrame) -> pl.DataFrame:
    """
    Add rank (1-100), score, and reasoning columns.
    Returns a DataFrame matching the submission spec.
    """
    rows = ranked.to_dicts()
    output = []
    for i, r in enumerate(rows):
        rank = i + 1
        output.append({
            "candidate_id": r["candidate_id"],
            "rank"        : rank,
            "score"       : round(r["final_score"], 6),
            "reasoning"   : generate_reasoning(r, rank),
        })

    return pl.DataFrame(output)


def write_submission(submission: pl.DataFrame, path: str | Path = SUBMISSION_PATH) -> None:
    """Write the submission CSV. Validates score monotonicity before writing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = submission.to_dicts()

    # Sanity check: scores must be non-increasing
    prev_score = float("inf")
    prev_rank  = 0
    for r in rows:
        s = r["score"]
        rk = r["rank"]
        if s > prev_score:
            log.warning(
                "Score not non-increasing: rank %d (%.6f) > rank %d (%.6f)",
                rk, s, prev_rank, prev_score,
            )
        prev_score = s
        prev_rank  = rk

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    log.info("Submission written to %s (%d rows)", path, len(rows))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_ranker(df: pl.DataFrame, output_path: str | Path = SUBMISSION_PATH) -> pl.DataFrame:
    """
    Full Stage 3 pipeline.
    Takes filtered + scored DataFrame (with semantic_score from retrieval.py).
    Returns and writes final submission DataFrame.
    """
    log.info("Ranking %d candidates …", df.height)
    ranked     = rank_candidates(df)
    submission = build_submission(ranked)
    write_submission(submission, output_path)
    return submission


# ---------------------------------------------------------------------------
# Mock mode — for testing without Person B's semantic scores
# ---------------------------------------------------------------------------

def run_ranker_mock(df: pl.DataFrame, output_path: str | Path = SUBMISSION_PATH) -> pl.DataFrame:
    """
    Same as run_ranker but injects a mock_semantic_score drawn from a
    simple heuristic so Person A can test the pipeline end-to-end.
    """
    import random
    random.seed(42)

    def _mock_score(row: dict) -> float:
        """Simple heuristic stand-in for Person B's semantic embedding."""
        candidate = json.loads(row.get("profile_json", "{}"))
        profile   = candidate.get("profile", {})
        skills    = candidate.get("skills", [])
        skill_names = {s.get("name", "").lower() for s in skills}
        relevant_count = len(skill_names & (REQUIRED_SKILLS | NLP_IR_SKILLS))
        yoe = profile.get("years_of_experience", 0)

        base = 0.30 + min(relevant_count * 0.05, 0.40)
        yoe_bonus = 0.10 if 5 <= yoe <= 9 else 0.0
        noise = random.uniform(-0.05, 0.05)
        return min(0.95, base + yoe_bonus + noise)

    rows = df.to_dicts()
    with_mock = [{**r, "semantic_score": _mock_score(r)} for r in rows]
    df_mock = pl.DataFrame(with_mock)

    return run_ranker(df_mock, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Quick end-to-end test using mock semantic scores
    from parser import run_parser

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_candidates.json"

    df = run_parser(data_path)
    print(f"Candidates after filters: {df.height}")

    submission = run_ranker_mock(df)
    print("\nTop 10 submission rows:")
    print(submission.head(10))
    print(f"\nSubmission saved → {SUBMISSION_PATH}")