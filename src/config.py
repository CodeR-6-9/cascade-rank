"""
config.py — Centralized thresholds, constants, and lookup tables.
All magic numbers live here. Change behaviour by editing this file only.
"""

from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT              = Path(__file__).resolve().parent.parent
DATA_DIR          = ROOT / "data"
CANDIDATES_JSONL  = DATA_DIR / "raw" / "candidates.jsonl"
CANDIDATES_GZ     = DATA_DIR / "raw" / "candidates.jsonl.gz"
SAMPLE_CANDIDATES = DATA_DIR / "raw" / "sample_candidates.json"
OUTPUT_DIR        = DATA_DIR / "output"
SUBMISSION_PATH   = OUTPUT_DIR / "final_submission.csv"

# ---------------------------------------------------------------------------
# JD metadata
# ---------------------------------------------------------------------------
JD_TITLE          = "Senior AI Engineer — Founding Team"
JD_MIN_YOE        = 4       # slightly below 5 — forgive 4yr if signals strong
JD_MAX_YOE        = 12
JD_TARGET_YOE_MIN = 5
JD_TARGET_YOE_MAX = 9

# ---------------------------------------------------------------------------
# Hard-disqualifier: pure consulting / services firms
# Disqualifies only if EVERY career role is at one of these.
# ---------------------------------------------------------------------------
PURE_SERVICES_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "hcl technologies",
    "tech mahindra", "mphasis", "hexaware", "niit technologies",
    "l&t infotech", "ltimindtree", "mindtree",
    "igate", "patni", "mastech", "syntel", "persistent systems", "zensar",
}

# ---------------------------------------------------------------------------
# Domain skills
# ---------------------------------------------------------------------------
WRONG_DOMAIN_SKILLS = {
    "computer vision", "cv", "image classification", "object detection",
    "yolo", "resnet", "image segmentation", "3d reconstruction",
    "speech recognition", "asr", "text-to-speech", "tts",
    "robotics", "ros", "slam", "autonomous driving", "lidar",
}

NLP_IR_SKILLS = {
    "nlp", "natural language processing", "text classification",
    "named entity recognition", "ner", "information retrieval",
    "semantic search", "embeddings", "sentence-transformers",
    "faiss", "pinecone", "qdrant", "weaviate", "milvus",
    "opensearch", "elasticsearch", "vector search", "vector database",
    "bm25", "tfidf", "ranking", "retrieval",
    "llm", "large language models", "transformers", "bert", "gpt",
    "fine-tuning", "fine-tuning llms", "rag", "retrieval augmented generation",
    "langchain", "haystack", "llamaindex",
}

REQUIRED_SKILLS = {
    "embeddings", "sentence-transformers", "faiss", "pinecone",
    "qdrant", "weaviate", "milvus", "opensearch", "elasticsearch",
    "vector search", "dense retrieval", "semantic search",
    "python", "pytorch", "tensorflow",
    "ndcg", "mrr", "map", "ranking evaluation", "a/b testing",
    "recommendation systems", "information retrieval",
}

HYPE_ONLY_SKILLS = {
    "langchain", "chatgpt", "openai api", "gpt wrapper",
    "prompt engineering", "llamaindex",
}

# ---------------------------------------------------------------------------
# Research-only titles (no production deployment)
# ---------------------------------------------------------------------------
RESEARCH_ONLY_TITLES = {
    "research scientist", "research engineer", "research intern",
    "postdoctoral researcher", "postdoc", "phd researcher",
    "research fellow", "academic researcher",
}

# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------
ALLOWED_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "delhi ncr",
    "gurugram", "gurgaon", "bengaluru", "bangalore", "chennai", "india",
}
PREFERRED_LOCATIONS = {"pune", "noida"}

# ---------------------------------------------------------------------------
# Behavioral signal thresholds
# ---------------------------------------------------------------------------
TODAY = date(2026, 6, 1)   # fixed reference date for reproducibility

GHOST_INACTIVITY_DAYS   = 180   # inactive > 6 months → ghost
STRONG_RECENCY_DAYS     = 30    # active in last 30d → bonus

IDEAL_NOTICE_DAYS       = 30
LONG_NOTICE_DAYS        = 60
HARD_NOTICE_CUTOFF      = 90

MIN_RESPONSE_RATE       = 0.10  # below this AND inactive → ghost (AND logic)
GOOD_RESPONSE_RATE      = 0.50
EXCELLENT_RESPONSE_RATE = 0.75

GOOD_GITHUB_SCORE       = 40
STRONG_GITHUB_SCORE     = 70

MIN_PROFILE_COMPLETENESS  = 50
GOOD_PROFILE_COMPLETENESS = 80

SALARY_MIN_REALISTIC    = 8
SALARY_MAX_REALISTIC    = 80

# ---------------------------------------------------------------------------
# Honeypot detection
# ---------------------------------------------------------------------------
MAX_EXPERT_ZERO_DURATION_SKILLS  = 4
MAX_YOE_CAREER_DISCREPANCY_YEARS = 3

# ---------------------------------------------------------------------------
# Ranker multiplier caps
# final_score = semantic × availability_mult × engagement_mult × fit_mult
# ---------------------------------------------------------------------------
AVAILABILITY_MAX = 1.25
AVAILABILITY_MIN = 0.30

ENGAGEMENT_MAX   = 1.30
ENGAGEMENT_MIN   = 0.70

FIT_MAX          = 1.15
FIT_MIN          = 0.50

# Boosts / penalties (additive on the 1.0 base within each multiplier)
BOOST_OPEN_TO_WORK         = +0.15
BOOST_RECENT_ACTIVE_30D    = +0.10
BOOST_GITHUB_GOOD          = +0.12
BOOST_GITHUB_STRONG        = +0.20
BOOST_SAVED_BY_RECRUITERS  = +0.08
BOOST_STRONG_RESPONSE_RATE = +0.05
BOOST_HIGH_INTERVIEW_RATE  = +0.05
BOOST_PREFERRED_LOCATION   = +0.10
BOOST_SHORT_NOTICE         = +0.08
BOOST_VERIFIED_CONTACT     = +0.05

PENALTY_LONG_NOTICE_60     = -0.10
PENALTY_LONG_NOTICE_90     = -0.20
PENALTY_GHOST_INACTIVE     = -0.35
PENALTY_LOW_RESPONSE       = -0.20
PENALTY_LOW_PROFILE        = -0.10
PENALTY_SALARY_MISMATCH    = -0.15
PENALTY_HYPE_SKILLS_ONLY   = -0.20

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
TOP_N           = 100
SUBMISSION_COLS = ["candidate_id", "rank", "score", "reasoning"]

# ---------------------------------------------------------------------------
# API CONTRACT  (Person A → Person B handoff)
# parser.py outputs a Polars DataFrame with these columns.
# retrieval.py adds 'semantic_score' (float 0–1).
# ranker.py reads the combined DataFrame.
# ---------------------------------------------------------------------------
PARSER_OUTPUT_COLUMNS = [
    "candidate_id",       # str
    "text_to_embed",      # str   rich text for sentence-transformers
    "years_of_experience",# float
    "signals_json",       # str   JSON-serialised redrob_signals dict
    "profile_json",       # str   JSON-serialised full candidate dict
    "honeypot_score",     # float 0–1
    "passed_hard_filters",# bool
    "filter_reason",      # str   '' if passed
]