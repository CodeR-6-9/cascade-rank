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

import os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
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
    if not career:
        return False
    for role in career:
        company = _n(role.get("company", ""))
        if not any(firm in company for firm in PURE_SERVICES_FIRMS):
            return False
    return True


def _is_research_only(career: list[dict]) -> bool:
    if not career:
        return False
    for role in career:
        industry = _n(role.get("industry", ""))
        title    = _n(role.get("title", ""))
        if industry not in ("academic", "research", "education", "university"):
            if not any(rt in title for rt in RESEARCH_ONLY_TITLES):
                return False
    return True


def _is_wrong_domain_only(skills: list[dict]) -> bool:
    names    = {_n(s.get("name", "")) for s in skills}
    has_bad  = names & WRONG_DOMAIN_SKILLS
    has_good = names & NLP_IR_SKILLS
    return bool(has_bad) and not has_good and len(has_bad) >= 3


def _has_no_relevant_skills(skills: list[dict], career: list[dict]) -> bool:
    """
    Disqualify if candidate has ZERO skills from NLP_IR_SKILLS | REQUIRED_SKILLS
    AND their career titles contain no AI/ML signal either.

    This catches Civil Engineers, Accountants, HR Managers etc. who slip through
    other filters because they have good behavioral signals (short notice, open to work)
    but zero technical relevance to the JD.

    We check career titles too so we don't drop someone with relevant experience
    who just hasn't listed skills properly.
    """
    skill_names = {_n(s.get("name", "")) for s in skills}
    if skill_names & (NLP_IR_SKILLS | REQUIRED_SKILLS):
        return False   # has at least one relevant skill → keep

    # Second chance: check career titles for AI/ML signal
    ai_title_keywords = {
        "machine learning", "ml engineer", "ai engineer", "data scientist",
        "nlp", "deep learning", "research scientist", "applied scientist",
        "recommendation", "search engineer", "ranking engineer",
        "applied ml", "computer vision",
    }
    for role in career:
        title = _n(role.get("title", ""))
        if any(kw in title for kw in ai_title_keywords):
            return False

    return True   # no relevant skills AND no relevant titles → disqualify


def _is_location_mismatch(profile: dict, signals: dict) -> bool:
    country = _n(profile.get("country", ""))
    willing = signals.get("willing_to_relocate", False)
    if country in ("india", "in", ""):
        return False
    return not willing


def _is_ghost(signals: dict) -> bool:
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
    score   = 0.0
    profile = candidate.get("profile", {})
    career  = candidate.get("career_history", [])
    skills  = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and s.get("duration_months", -1) == 0
    )
    if expert_zero > MAX_EXPERT_ZERO_DURATION_SKILLS:
        score += 0.40

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

    expert_count = sum(1 for s in skills if s.get("proficiency") in ("expert", "advanced"))
    if len(skills) > 25 and expert_count > 15:
        score += 0.25

    title = _n(profile.get("current_title", ""))
    non_tech = {"marketing manager", "sales manager", "operations manager",
                "hr manager", "finance manager", "accountant", "customer support"}
    if title in non_tech:
        ml_skills = {_n(s.get("name", "")) for s in skills} & (REQUIRED_SKILLS | NLP_IR_SKILLS)
        if len(ml_skills) >= 5:
            score += 0.30

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
    rows = []
    for c in candidates:
        cid     = c.get("candidate_id", "")
        profile = c.get("profile", {})
        career  = c.get("career_history", [])
        skills  = c.get("skills", [])
        signals = c.get("redrob_signals", {})

        passed = True
        reason = ""

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
        elif _has_no_relevant_skills(skills, career):
            passed, reason = False, "no_relevant_ai_ml_skills"

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

    # Log drop reasons at INFO level so we can always see what's being filtered
    reason_counts = (
        df.filter(~pl.col("passed_hard_filters"))
        .group_by("filter_reason")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    log.info("Drop reasons:\n%s", reason_counts)

    return passing


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    use_sample = "--sample" in sys.argv
    df = run_parser(use_sample=use_sample)
    print(f"\nSurviving candidates: {df.height}")
    print(df.select(["candidate_id", "years_of_experience", "honeypot_score"]).head(10))