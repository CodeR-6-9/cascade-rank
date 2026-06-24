"""
config.py — Centralized thresholds, constants, and lookup tables.
All magic numbers live here. Change behaviour by editing this file only.
"""

from datetime import date

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CANDIDATES_GZ   = "data/raw/candidates.jsonl.gz"
CANDIDATES_JSONL = "data/raw/candidates.jsonl"
SAMPLE_CANDIDATES = "data/raw/sample_candidates.json"
OUTPUT_DIR      = "data/output"
SUBMISSION_PATH = "data/output/final_submission.csv"

# ---------------------------------------------------------------------------
# JD metadata
# ---------------------------------------------------------------------------
JD_TITLE = "Senior AI Engineer — Founding Team"
JD_MIN_YOE = 4          # we accept slightly below 5 if other signals strong
JD_MAX_YOE = 12         # beyond 12 years is not a hard disqualifier but less ideal
JD_TARGET_YOE_MIN = 5
JD_TARGET_YOE_MAX = 9

# ---------------------------------------------------------------------------
# Hard-disqualifier: consulting / services-only firms
# A candidate is disqualified only if EVERY role in their career is at one
# of these firms (no product company experience at all).
# ---------------------------------------------------------------------------
PURE_SERVICES_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "hcl technologies",
    "tech mahindra", "mphasis", "hexaware", "niit technologies",
    "l&t infotech", "ltimindtree", "mindtree",
    "igate", "patni", "mastech", "syntel",
}

# ---------------------------------------------------------------------------
# Hard-disqualifier: wrong domain (primary skills)
# Candidate is flagged if their top-3 skills (by endorsements) are all
# from this set AND they have no NLP/IR skills at all.
# ---------------------------------------------------------------------------
WRONG_DOMAIN_SKILLS = {
    "computer vision", "cv", "image classification", "object detection",
    "yolo", "resnet", "image segmentation", "3d reconstruction",
    "speech recognition", "asr", "text-to-speech", "tts",
    "robotics", "ros", "slam", "autonomous driving", "lidar",
    "openCV",  # fine as secondary skill but not primary
}

# NLP / IR skills — presence of any of these redeems a CV candidate
NLP_IR_SKILLS = {
    "nlp", "natural language processing", "text classification",
    "named entity recognition", "ner", "information retrieval",
    "semantic search", "embeddings", "sentence-transformers",
    "faiss", "pinecone", "qdrant", "weaviate", "milvus",
    "opensearch", "elasticsearch", "vector search", "vector database",
    "bm25", "tfidf", "ranking", "retrieval",
    "llm", "large language model", "transformers", "bert", "gpt",
    "fine-tuning", "fine-tuning llms", "rag", "retrieval augmented",
    "langchain", "haystack", "llamaindex",
}

# Core required skills from JD — used for semantic scoring boost
REQUIRED_SKILLS = {
    "embeddings", "sentence-transformers", "faiss", "pinecone",
    "qdrant", "weaviate", "milvus", "opensearch", "elasticsearch",
    "vector search", "dense retrieval", "semantic search",
    "python", "pytorch", "tensorflow",
    "ndcg", "mrr", "map", "ranking evaluation", "a/b testing",
    "recommendation systems", "information retrieval",
}

BONUS_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning llms", "fine-tuning",
    "learning to rank", "xgboost", "lightgbm",
    "distributed systems", "kafka", "spark",
}

# LangChain-only "recent AI" keywords — not disqualifying alone, but
# combined with short AI tenure and no production signals → downweight
HYPE_ONLY_SKILLS = {
    "langchain", "chatgpt", "openai api", "gpt wrapper",
    "prompt engineering", "llamaindex",
}

# ---------------------------------------------------------------------------
# Hard-disqualifier: research-only titles (no product deployment)
# ---------------------------------------------------------------------------
RESEARCH_ONLY_TITLES = {
    "research scientist", "research engineer", "research intern",
    "postdoctoral researcher", "postdoc", "phd researcher",
    "research fellow", "academic researcher",
}

# ---------------------------------------------------------------------------
# Location — allowed cities / regions for this JD
# ---------------------------------------------------------------------------
ALLOWED_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "delhi ncr",
    "gurugram", "gurgaon", "bengaluru", "bangalore", "chennai",
    "india",   # catch-all for Indian candidates
}

PREFERRED_LOCATIONS = {"pune", "noida"}  # JD explicitly prefers these

# ---------------------------------------------------------------------------
# Behavioral signal thresholds
# ---------------------------------------------------------------------------
TODAY = date.today()

# Recency
GHOST_INACTIVITY_DAYS    = 180   # inactive > 6 months → ghost penalty
STRONG_RECENCY_DAYS      = 30    # active in last 30d → bonus

# Notice period
IDEAL_NOTICE_DAYS        = 30    # JD says prefer sub-30
BUYOUT_NOTICE_DAYS       = 30    # company can buy out up to 30 days
LONG_NOTICE_DAYS         = 60    # >60 days → soft penalty
HARD_NOTICE_CUTOFF       = 90    # >90 days → stronger penalty (not hard disqualifier)

# Engagement
MIN_RESPONSE_RATE        = 0.15  # below this → ghost
GOOD_RESPONSE_RATE       = 0.50
EXCELLENT_RESPONSE_RATE  = 0.75

GOOD_GITHUB_SCORE        = 40    # 0-100
STRONG_GITHUB_SCORE      = 70

MIN_PROFILE_COMPLETENESS = 50    # below this → penalty
GOOD_PROFILE_COMPLETENESS = 80

# Salary (INR LPA) — rough market range for this role in India
SALARY_MIN_REALISTIC     = 8     # below this is suspiciously low
SALARY_MAX_REALISTIC     = 80    # above this signals mismatch with a Series A

# ---------------------------------------------------------------------------
# Honeypot detection
# ---------------------------------------------------------------------------
# Expert/advanced skill with 0 months duration is a red flag
MAX_EXPERT_ZERO_DURATION_SKILLS = 4   # >4 such skills → suspect
# experience years claimed vs career dates mismatch
MAX_YOE_CAREER_DISCREPANCY_YEARS = 3

# ---------------------------------------------------------------------------
# Scoring weights for ranker.py
# Final score = semantic_score × availability_mult × engagement_mult × fit_mult
# ---------------------------------------------------------------------------
WEIGHT_SEMANTIC      = 1.0   # base — provided by Person B

# Availability multiplier caps
AVAILABILITY_MAX     = 1.25
AVAILABILITY_MIN     = 0.30

# Engagement multiplier caps
ENGAGEMENT_MAX       = 1.30
ENGAGEMENT_MIN       = 0.70

# Fit multiplier caps
FIT_MAX              = 1.15
FIT_MIN              = 0.50

# Individual boosts / penalties (additive on top of 1.0 base)
BOOST_OPEN_TO_WORK          = +0.15
BOOST_RECENT_ACTIVE_30D     = +0.10
BOOST_GITHUB_GOOD           = +0.12
BOOST_GITHUB_STRONG         = +0.20
BOOST_SAVED_BY_RECRUITERS   = +0.08   # saved_by_recruiters_30d > 3
BOOST_STRONG_RESPONSE_RATE  = +0.05
BOOST_HIGH_INTERVIEW_RATE   = +0.05   # interview_completion_rate > 0.8
BOOST_PREFERRED_LOCATION    = +0.10
BOOST_SHORT_NOTICE          = +0.08   # <= 30 days
BOOST_VERIFIED_CONTACT      = +0.05   # both email + phone verified

PENALTY_LONG_NOTICE_60      = -0.10
PENALTY_LONG_NOTICE_90      = -0.20
PENALTY_GHOST_INACTIVE      = -0.35
PENALTY_LOW_RESPONSE        = -0.20
PENALTY_LOW_PROFILE         = -0.10
PENALTY_SALARY_MISMATCH     = -0.15
PENALTY_HYPE_SKILLS_ONLY    = -0.20   # LangChain-only AI experience

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
TOP_N = 100   # submission requires exactly 100 candidates