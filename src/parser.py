"""
parser.py — Stage 1: Load candidates and apply deterministic trap filters.

Pipeline:
  load_candidates()  →  list[dict]   (all candidates from jsonl)
  apply_filters()    →  pl.DataFrame (surviving candidates only)
  _build_embed_text()→  text for Person B's embedding model

Output DataFrame columns (API contract with Person B):
  candidate_id, text_to_embed, years_of_experience,
  signals_json, profile_json, honeypot_score,
  passed_hard_filters, filter_reason

Run standalone:
  python -m src.parser                    # full 100K run
  python -m src.parser --sample           # 50-record sample, fast
"""

from __future__ import annotations

import gzip
import json
import logging
import sys
from datetime import date
from pathlib import Path

import orjson
import polars as pl

# ── config import ──────────────────────────────────────────────────────────────
# Works whether called as `python -m src.parser` or `python src/parser.py`
import os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    PURE_SERVICES_FIRMS, WRONG_DOMAIN_SKILLS, NLP_IR_SKILLS,
    REQUIRED_SKILLS, HYPE_ONLY_SKILLS, RESEARCH_ONLY_TITLES,
    ALLOWED_LOCATIONS, TODAY, GHOST_INACTIVITY_DAYS, MIN_RESPONSE_RATE,
    MAX_EXPERT_ZERO_DURATION_SKILLS, JD_MIN_YOE,
    SAMPLE_CANDIDATES, CANDIDATES_JSONL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [parser] %(message)s")
log = logging.getLogger(__name__)


# ── Loading ────────────────────────────────────────────────────────────────────

def load_candidates(path: Path) -> list[dict]:
    """Stream-load .jsonl or .jsonl.gz — memory efficient for 100K records."""
    log.info("Loading candidates from %s", path)
    opener = gzip.open if path.suffix == ".gz" else open
    mode   = "rb" if path.suffix == ".gz" else "r"
    out: list[dict] = []
    with opener(path, mode) as f:
        for line in f:
            if line.strip():
                out.append(orjson.loads(line))
    log.info("Loaded %d candidates", len(out))
    return out


def load_sample(path: Path) -> list[dict]:
    with open(path, "rb") as f:
        return orjson.loads(f.read())


# ── Filter helpers (True = KEEP, False = DROP) ─────────────────────────────────

def _n(s: str) -> str:
    return s.strip().lower()


def _is_consulting_only(career: list[dict]) -> bool:
    """Disqualify only if EVERY role is at a known services firm."""
    if not career:
        return False
    for role in career:
        company = _n(role.get("company", ""))
        if not any(firm in company for firm in PURE_SERVICES_FIRMS):
            return False   # found a non-consulting company → keep
    return True


def _is_research_only(career: list[dict]) -> bool:
    """Disqualify if all roles are academic/research with no product deployment."""
    if not career:
        return False
    for role in career:
        industry = _n(role.get("industry", ""))
        title    = _n(role.get("title", ""))
        if industry not in ("academic", "research", "education", "university"):
            if not any(rt in title for rt in RESEARCH_ONLY_TITLES):
                return False   # product role found → keep
    return True


def _is_wrong_domain_only(skills: list[dict]) -> bool:
    """CV/Speech/Robotics only with zero NLP/IR → disqualify."""
    names    = {_n(s.get("name", "")) for s in skills}
    has_bad  = names & WRONG_DOMAIN_SKILLS
    has_good = names & NLP_IR_SKILLS
    return bool(has_bad) and not has_good and len(has_bad) >= 3


def _is_location_mismatch(profile: dict, signals: dict) -> bool:
    """Outside India AND unwilling to relocate → disqualify."""
    country = _n(profile.get("country", ""))
    willing = signals.get("willing_to_relocate", False)
    if country in ("india", "in", ""):
        return False
    return not willing


def _is_ghost(signals: dict) -> bool:
    """
    Ghost = inactive > 6 months AND not open-to-work AND response rate < 10%.
    Using AND logic: someone who forgot to log in but still replies = keep.
    """
    last_str = signals.get("last_active_date", "")
    try:
        days_inactive = (TODAY - date.fromisoformat(last_str)).days
    except (ValueError, TypeError):
        days_inactive = 999

    open_to_work  = signals.get("open_to_work_flag", False)
    response_rate = signals.get("recruiter_response_rate", 0.0)

    return (
        days_inactive > GHOST_INACTIVITY_DAYS
        and not open_to_work
        and response_rate < MIN_RESPONSE_RATE
    )


# ── Honeypot scoring ──────────────────────────────────────────────────────────

def _honeypot_score(candidate: dict) -> float:
    """
    Soft suspicion score 0–1. Not a hard disqualifier — the ranker
    applies this as a multiplier penalty.
    """
    score   = 0.0
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
        score += 0.40

    # 2. Claimed YOE vs actual career start dates
    claimed_yoe = float(profile.get("years_of_experience", 0))
    earliest_start = None
    for role in career:
        try:
            sd = date.fromisoformat(role["start_date"])
            if earliest_start is None or sd < earliest_start:
                earliest_start = sd
        except (KeyError, ValueError):
            pass
    if earliest_start:
        actual_yoe = (TODAY - earliest_start).days / 365.25
        if claimed_yoe > actual_yoe + 3:
            score += 0.35

    # 3. Too many skills with uniformly high proficiency
    expert_count = sum(1 for s in skills if s.get("proficiency") in ("expert", "advanced"))
    if len(skills) > 25 and expert_count > 15:
        score += 0.25

    # 4. Non-tech title + many ML skills = contradictory profile
    title = _n(profile.get("current_title", ""))
    non_tech = {"marketing manager", "sales manager", "operations manager",
                "hr manager", "finance manager", "accountant", "customer support"}
    if title in non_tech:
        ml_skills = {_n(s.get("name", "")) for s in skills} & (REQUIRED_SKILLS | NLP_IR_SKILLS)
        if len(ml_skills) >= 5:
            score += 0.30

    # 5. Perfect completeness + 2+ years inactive
    completeness = signals.get("profile_completeness_score", 0)
    last_str     = signals.get("last_active_date", "")
    try:
        days_inactive = (TODAY - date.fromisoformat(last_str)).days
    except (ValueError, TypeError):
        days_inactive = 0
    if completeness == 100 and days_inactive > 730:
        score += 0.15

    return min(score, 1.0)


# ── Text construction for embedding ───────────────────────────────────────────

def _build_embed_text(c: dict) -> str:
    """
    Dense text string for sentence-transformers.
    Headline + summary + top-3 career descriptions + skills by proficiency.
    """
    parts: list[str] = []
    profile = c.get("profile", {})

    parts.append(profile.get("headline", ""))
    parts.append(profile.get("summary", ""))

    for role in c.get("career_history", [])[:3]:
        title   = role.get("title", "")
        company = role.get("company", "")
        desc    = role.get("description", "")
        parts.append(f"{title} at {company}: {desc}")

    skill_lines = []
    for level in ("expert", "advanced", "intermediate"):
        for s in c.get("skills", []):
            if s.get("proficiency") == level:
                name   = s.get("name", "")
                months = s.get("duration_months", 0)
                skill_lines.append(f"{name} ({months}m)")
    parts.append("Skills: " + ", ".join(skill_lines))

    for edu in c.get("education", [])[:2]:
        parts.append(
            f"{edu.get('degree', '')} in {edu.get('field_of_study', '')} "
            f"from {edu.get('institution', '')}"
        )

    return " | ".join(p for p in parts if p.strip())


# ── Core filter pipeline ───────────────────────────────────────────────────────

def apply_filters(candidates: list[dict]) -> pl.DataFrame:
    """
    Run all hard filters. Returns Polars DataFrame with all candidates
    (passed_hard_filters = True/False). Use run_parser() to get only passing rows.
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

        # Cheapest checks first
        yoe = float(profile.get("years_of_experience", 0))
        if yoe < JD_MIN_YOE:
            passed, reason = False, "insufficient_yoe"
        elif _is_ghost(signals):
            passed, reason = False, "ghost_candidate"
        elif _is_location_mismatch(profile, signals):
            passed, reason = False, "location_mismatch"
        elif _is_consulting_only(career):
            passed, reason = False, "consulting_only"
        elif _is_research_only(career):
            passed, reason = False, "research_only"
        elif _is_wrong_domain_only(skills):
            passed, reason = False, "wrong_domain_cv_speech_robotics"

        hp = _honeypot_score(c)

        rows.append({
            "candidate_id":        cid,
            "passed_hard_filters": passed,
            "filter_reason":       reason,
            "honeypot_score":      hp,
            "years_of_experience": yoe,
            "text_to_embed":       _build_embed_text(c),
            "signals_json":        json.dumps(signals),
            "profile_json":        json.dumps(c),
        })

    df = pl.DataFrame(rows)
    n_total  = len(df)
    n_passed = df.filter(pl.col("passed_hard_filters")).height
    log.info(
        "Filter complete — passed: %d / %d (%.1f%%)",
        n_passed, n_total, 100 * n_passed / max(n_total, 1),
    )
    return df


# ── Public API ─────────────────────────────────────────────────────────────────

def run_parser(candidates_path: Path | str | None = None, use_sample: bool = False) -> pl.DataFrame:
    """
    Full Stage 1 pipeline. Returns only the passing candidates.

    Args:
        candidates_path: explicit path override (optional)
        use_sample:      use sample_candidates.json (50 records, fast)
    """
    if candidates_path is not None:
        path = Path(candidates_path)
    elif use_sample:
        path = SAMPLE_CANDIDATES
    else:
        path = CANDIDATES_JSONL

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix == ".json":
        candidates = load_sample(path)
    else:
        candidates = load_candidates(path)

    df = apply_filters(candidates)
    passing = df.filter(pl.col("passed_hard_filters"))

    if log.isEnabledFor(logging.DEBUG):
        reason_counts = (
            df.filter(~pl.col("passed_hard_filters"))
            .group_by("filter_reason")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        log.debug("Drop reasons:\n%s", reason_counts)

    return passing


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    use_sample = "--sample" in sys.argv
    df = run_parser(use_sample=use_sample)
    print(f"\nSurviving candidates: {df.height}")
    print(df.select(["candidate_id", "years_of_experience", "honeypot_score"]).head(10))