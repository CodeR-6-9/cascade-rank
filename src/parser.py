"""
parser.py — Stage 1: Load candidates and apply deterministic trap filters.

Pipeline:
  load_candidates()  →  DataFrame (all 100K candidates)
  apply_filters()    →  DataFrame (surviving candidates only)
  build_text()       →  adds `text_to_embed` column for Person B's retrieval step
  detect_honeypots() →  flags subtly impossible profiles

Output DataFrame columns (the API contract with Person B):
  candidate_id, text_to_embed, years_of_experience, signals_dict,
  passed_hard_filters, filter_reason, profile_json
"""

from __future__ import annotations

import gzip
import json
import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import orjson
import polars as pl

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PURE_SERVICES_FIRMS, WRONG_DOMAIN_SKILLS, NLP_IR_SKILLS,
    REQUIRED_SKILLS, HYPE_ONLY_SKILLS, RESEARCH_ONLY_TITLES,
    ALLOWED_LOCATIONS, TODAY, GHOST_INACTIVITY_DAYS,
    MAX_EXPERT_ZERO_DURATION_SKILLS, JD_MIN_YOE,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_candidates(path: str | Path) -> list[dict]:
    """Load candidates from .jsonl or .jsonl.gz — returns list of dicts."""
    path = Path(path)
    log.info("Loading candidates from %s", path)

    opener = gzip.open if path.suffix == ".gz" else open
    mode   = "rb" if path.suffix == ".gz" else "r"
    candidates: list[dict] = []

    with opener(path, mode) as f:
        for line in f:
            if line.strip():
                candidates.append(orjson.loads(line))

    log.info("Loaded %d candidates", len(candidates))
    return candidates


def load_sample(path: str | Path) -> list[dict]:
    """Load the 50-candidate sample JSON (pretty-printed array)."""
    with open(path, "rb") as f:
        return orjson.loads(f.read())


# ---------------------------------------------------------------------------
# Individual filter helpers  (all return True = KEEP, False = DROP)
# ---------------------------------------------------------------------------

def _normalise(s: str) -> str:
    return s.strip().lower()


def _is_consulting_only(career: list[dict]) -> bool:
    """True if every role is at a pure services firm → disqualify."""
    if not career:
        return False
    product_company_found = False
    for role in career:
        company = _normalise(role.get("company", ""))
        # Check if any token in the company name is a known services firm
        if not any(firm in company for firm in PURE_SERVICES_FIRMS):
            product_company_found = True
            break
    return not product_company_found


def _is_research_only(career: list[dict]) -> bool:
    """True if ALL roles are research-only with no product deployment."""
    if not career:
        return False
    for role in career:
        title = _normalise(role.get("title", ""))
        industry = _normalise(role.get("industry", ""))
        # any industry other than "academia" / "research" counts as product
        if industry not in ("academic", "research", "education", "university"):
            # double-check title isn't PURELY research
            if not any(rt in title for rt in RESEARCH_ONLY_TITLES):
                return False
            # title is research-y but industry is product → keep
            if industry not in ("academic", "research", "education"):
                return False
    return True


def _skill_names(skills: list[dict]) -> list[str]:
    return [_normalise(s.get("name", "")) for s in skills]


def _is_wrong_domain_only(skills: list[dict]) -> bool:
    """True if candidate's domain is purely CV/Speech/Robotics with no NLP/IR."""
    names = set(_skill_names(skills))
    has_wrong = names & WRONG_DOMAIN_SKILLS
    has_nlp   = names & NLP_IR_SKILLS
    if not has_wrong:
        return False
    # Only penalise if NLP/IR is COMPLETELY absent
    return len(has_nlp) == 0 and len(has_wrong) >= 3


def _is_location_mismatch(profile: dict, signals: dict) -> bool:
    """True if location is outside India AND candidate won't relocate."""
    country  = _normalise(profile.get("country", ""))
    location = _normalise(profile.get("location", ""))
    willing  = signals.get("willing_to_relocate", False)

    if country == "india" or country == "in":
        return False  # India-based → always fine
    if willing:
        return False  # willing to relocate → fine
    # Foreign candidate who won't relocate
    if not any(city in location for city in ALLOWED_LOCATIONS):
        return True
    return False


def _is_ghost(signals: dict) -> bool:
    """True if candidate is effectively unreachable."""
    last_active_str = signals.get("last_active_date", "")
    try:
        last_active = date.fromisoformat(last_active_str)
        days_inactive = (TODAY - last_active).days
    except (ValueError, TypeError):
        days_inactive = 999

    open_to_work   = signals.get("open_to_work_flag", False)
    response_rate  = signals.get("recruiter_response_rate", 0.0)

    # Ghost: very long inactive AND not open to work AND very low response
    if days_inactive > GHOST_INACTIVITY_DAYS and not open_to_work and response_rate < 0.10:
        return True
    return False


def _is_experience_too_low(profile: dict) -> bool:
    yoe = profile.get("years_of_experience", 0)
    return yoe < JD_MIN_YOE


def _honeypot_score(candidate: dict) -> float:
    """
    Return a suspicion score 0.0–1.0 for honeypot likelihood.
    0.0 = clean, 1.0 = almost certainly a honeypot.
    """
    score = 0.0
    profile = candidate.get("profile", {})
    career  = candidate.get("career_history", [])
    skills  = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    # 1. Expert/advanced skills with zero duration — classic stuffing
    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and s.get("duration_months", -1) == 0
    )
    if expert_zero > MAX_EXPERT_ZERO_DURATION_SKILLS:
        score += 0.4

    # 2. Claimed YOE vs career dates — can't have 8 yrs exp at a 3-yr-old company
    claimed_yoe = profile.get("years_of_experience", 0)
    if career:
        earliest_start = None
        for role in career:
            try:
                sd = date.fromisoformat(role["start_date"])
                if earliest_start is None or sd < earliest_start:
                    earliest_start = sd
            except (KeyError, ValueError):
                pass
        if earliest_start:
            actual_years = (TODAY - earliest_start).days / 365.25
            if claimed_yoe > actual_years + 3:   # claimed 3+ years more than dates allow
                score += 0.35

    # 3. Too many skills (>25) with uniformly high proficiency
    expert_count = sum(1 for s in skills if s.get("proficiency") in ("expert", "advanced"))
    if len(skills) > 25 and expert_count > 15:
        score += 0.25

    # 4. Contradictory current role: title is "Marketing Manager" but skills are all ML
    title = _normalise(profile.get("current_title", ""))
    non_tech_titles = {"marketing manager", "sales manager", "operations manager",
                       "hr manager", "finance manager", "accountant", "customer support"}
    if title in non_tech_titles:
        ml_skills = set(_skill_names(skills)) & (REQUIRED_SKILLS | NLP_IR_SKILLS)
        if len(ml_skills) >= 5:
            score += 0.30

    # 5. Profile completeness 100 but last_active 2+ years ago
    completeness = signals.get("profile_completeness_score", 0)
    last_active_str = signals.get("last_active_date", "")
    try:
        last_active = date.fromisoformat(last_active_str)
        days_inactive = (TODAY - last_active).days
    except (ValueError, TypeError):
        days_inactive = 0

    if completeness == 100 and days_inactive > 730:
        score += 0.15

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Core filter pipeline
# ---------------------------------------------------------------------------

def apply_filters(candidates: list[dict]) -> pl.DataFrame:
    """
    Run all hard filters and return a Polars DataFrame with columns:
      candidate_id, passed_hard_filters, filter_reason, honeypot_score,
      years_of_experience, text_to_embed, signals_dict, profile_json
    """
    rows = []
    for c in candidates:
        cid     = c.get("candidate_id", "")
        profile = c.get("profile", {})
        career  = c.get("career_history", [])
        skills  = c.get("skills", [])
        signals = c.get("redrob_signals", {})

        passed = True
        reason = ""

        # --- Hard disqualifiers (order matters — cheapest checks first) ---

        if _is_experience_too_low(profile):
            passed, reason = False, "insufficient_yoe"

        elif _is_ghost(signals):
            passed, reason = False, "ghost_candidate"

        elif _is_location_mismatch(profile, signals):
            passed, reason = False, "location_mismatch"

        elif _is_consulting_only(career):
            passed, reason = False, "consulting_only"

        elif _is_wrong_domain_only(skills):
            passed, reason = False, "wrong_domain_cv_speech_robotics"

        # Honeypot check (soft — we flag but don't hard-drop;
        # the ranker will down-weight strongly)
        hp_score = _honeypot_score(c)

        rows.append({
            "candidate_id"       : cid,
            "passed_hard_filters": passed,
            "filter_reason"      : reason,
            "honeypot_score"     : hp_score,
            "years_of_experience": float(profile.get("years_of_experience", 0)),
            "text_to_embed"      : _build_embed_text(c),
            "signals_json"       : json.dumps(signals),   # serialised for polars
            "profile_json"       : json.dumps(c),          # full candidate for ranker
        })

    df = pl.DataFrame(rows)
    n_total    = len(df)
    n_passed   = df.filter(pl.col("passed_hard_filters")).height
    log.info(
        "Filter pass rate: %d / %d  (%.1f%%)",
        n_passed, n_total, 100 * n_passed / max(n_total, 1),
    )
    return df


# ---------------------------------------------------------------------------
# Text construction for embedding (Person B's input)
# ---------------------------------------------------------------------------

def _build_embed_text(c: dict) -> str:
    """
    Build a single string that captures the candidate's most relevant
    content for semantic similarity against the JD.

    Strategy: headline + summary + top-3 career descriptions (most recent)
              + skill names weighted by proficiency.
    """
    parts: list[str] = []
    profile = c.get("profile", {})

    # 1. Headline and summary
    parts.append(profile.get("headline", ""))
    parts.append(profile.get("summary", ""))

    # 2. Career descriptions — most recent 3 roles
    for role in c.get("career_history", [])[:3]:
        title = role.get("title", "")
        company = role.get("company", "")
        desc = role.get("description", "")
        parts.append(f"{title} at {company}: {desc}")

    # 3. Skills — expert/advanced first, then intermediate
    skill_lines = []
    for level in ("expert", "advanced", "intermediate"):
        for s in c.get("skills", []):
            if s.get("proficiency") == level:
                name = s.get("name", "")
                months = s.get("duration_months", 0)
                skill_lines.append(f"{name} ({months}m)")
    parts.append("Skills: " + ", ".join(skill_lines))

    # 4. Education field of study
    for edu in c.get("education", [])[:2]:
        parts.append(f"{edu.get('degree', '')} in {edu.get('field_of_study', '')} "
                     f"from {edu.get('institution', '')}")

    return " | ".join(filter(None, parts))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_parser(candidates_path: str | Path) -> pl.DataFrame:
    """
    Full Stage 1 pipeline.
    Returns filtered DataFrame ready to pass to retrieval.py.
    """
    path = Path(candidates_path)
    if path.suffix == ".json":
        candidates = load_sample(path)
    else:
        candidates = load_candidates(path)

    df = apply_filters(candidates)

    # Separate passing from failing for logging
    passing = df.filter(pl.col("passed_hard_filters"))
    failing = df.filter(~pl.col("passed_hard_filters"))

    if log.isEnabledFor(logging.DEBUG):
        reason_counts = (
            failing
            .group_by("filter_reason")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        log.debug("Drop reasons:\n%s", reason_counts)

    return passing


# ---------------------------------------------------------------------------
# CLI  (quick test / dev)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_candidates.json"
    df = run_parser(path)

    print(f"\nSurviving candidates: {df.height}")
    print(df.select(["candidate_id", "years_of_experience", "honeypot_score"]).head(10))