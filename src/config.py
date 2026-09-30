"""
Central configuration for the AI Enterprise Knowledge Assistant.

Configuration can be loaded from:
1. Streamlit Secrets when deployed on Streamlit Cloud
2. .env when running locally

This allows the same codebase to work both locally and in production.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

# src/config.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Local .env support
# ---------------------------------------------------------------------------

# Load .env when running locally.
# If Streamlit Secrets are available, they will take precedence below.
load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _get_str(name: str, default: str) -> str:
    """
    Get a string configuration value.

    Priority:
    1. Streamlit Secrets
    2. Environment variable / .env
    3. Default value
    """
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        # Allows the module to work outside Streamlit.
        pass

    return os.getenv(name, default)


def _get_int(name: str, default: int) -> int:
    """
    Get an integer configuration value.

    Priority:
    1. Streamlit Secrets
    2. Environment variable / .env
    3. Default value
    """
    value = _get_str(name, str(default))

    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _get_float(name: str, default: float) -> float:
    """
    Get a float configuration value.

    Priority:
    1. Streamlit Secrets
    2. Environment variable / .env
    3. Default value
    """
    value = _get_str(name, str(default))

    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Groq LLM configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY = _get_str("GROQ_API_KEY", "")

GROQ_MODEL = _get_str(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile",
)


# ---------------------------------------------------------------------------
# Embedding configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = _get_str(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)


# ---------------------------------------------------------------------------
# Retrieval configuration
# ---------------------------------------------------------------------------

TOP_K = _get_int("TOP_K", 5)

SIMILARITY_THRESHOLD = _get_float(
    "SIMILARITY_THRESHOLD",
    0.35,
)


# ---------------------------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------------------------

CHUNK_SIZE = _get_int("CHUNK_SIZE", 800)

CHUNK_OVERLAP = _get_int(
    "CHUNK_OVERLAP",
    100,
)


# ---------------------------------------------------------------------------
# Domains supported by the MVP
# ---------------------------------------------------------------------------

DOMAINS = [
    "HR",
    "TECHNICAL",
    "PROJECTS",
    "GENERAL",
]


# ---------------------------------------------------------------------------
# Domain folder mapping
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# No-answer message
# ---------------------------------------------------------------------------

NO_ANSWER_MESSAGE = (
    "I couldn't find enough information in the company documents "
    "to answer this question."
)


# ---------------------------------------------------------------------------
# Index paths
# ---------------------------------------------------------------------------

def index_paths(domain: str):
    """
    Return (faiss_index_path, metadata_path) for a given domain.
    """

    folder = DOMAIN_FOLDER.get(domain, "general")

    domain_dir = INDEX_DIR / folder

    return (
        domain_dir / "index.faiss",
        domain_dir / "metadata.pkl",
    )