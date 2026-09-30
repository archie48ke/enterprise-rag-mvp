"""
Central configuration for the AI Enterprise Knowledge Assistant.

All tunable values are loaded from environment variables (.env) so that
nothing is hard-coded, and so a student can retune the system (chunk size,
similarity threshold, top_k, etc.) without touching any code.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root, regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _get_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Groq LLM configuration
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = _get_str("GROQ_MODEL", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------------------
# Embedding configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = _get_str("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Retrieval configuration
# ---------------------------------------------------------------------------
TOP_K = _get_int("TOP_K", 5)

# NOTE: This threshold is a reasonable STARTING POINT only. It is not a
# universally correct value. Cosine-similarity thresholds are sensitive to
# the embedding model, the domain vocabulary, and chunk size, so it should
# be tuned empirically against your own documents and test questions.
SIMILARITY_THRESHOLD = _get_float("SIMILARITY_THRESHOLD", 0.35)

# ---------------------------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------------------------
CHUNK_SIZE = _get_int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP", 100)

# ---------------------------------------------------------------------------
# Domains supported by the MVP
# ---------------------------------------------------------------------------
DOMAINS = ["HR", "TECHNICAL", "PROJECTS", "GENERAL"]

# Maps a domain name to its data / index folder name on disk.
DOMAIN_FOLDER = {
    "HR": "hr",
    "TECHNICAL": "technical",
    "PROJECTS": "projects",
    "GENERAL": "general",
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "indexes"

NO_ANSWER_MESSAGE = (
    "I couldn't find enough information in the company documents "
    "to answer this question."
)


def index_paths(domain: str):
    """Return (faiss_index_path, metadata_path) for a given domain."""
    folder = DOMAIN_FOLDER.get(domain, "general")
    domain_dir = INDEX_DIR / folder
    return domain_dir / "index.faiss", domain_dir / "metadata.pkl"
