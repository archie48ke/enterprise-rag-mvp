"""
Tests for the Manager Agent (routing) and the end-to-end LangGraph workflow.

These tests call the real Groq API, so they are skipped automatically if
GROQ_API_KEY is not configured. They also assume the sample indexes have
been built (see scripts/generate_sample_pdfs.py + scripts/build_indexes.py).

Run with:
    pytest tests/test_workflow.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GROQ_API_KEY, INDEX_DIR, DOMAIN_FOLDER, NO_ANSWER_MESSAGE
from src.agents import classify_domain
from src.workflow import run_workflow

requires_groq = pytest.mark.skipif(
    not GROQ_API_KEY, reason="GROQ_API_KEY not configured"
)


def _all_indexes_missing() -> bool:
    return not any(
        (INDEX_DIR / folder / "index.faiss").exists()
        for folder in DOMAIN_FOLDER.values()
    )


requires_indexes = pytest.mark.skipif(
    _all_indexes_missing(), reason="No FAISS indexes built yet"
)


@requires_groq
def test_router_hr_question():
    assert classify_domain("How many casual leaves do I get?") == "HR"


@requires_groq
def test_router_technical_question():
    assert classify_domain("How do I deploy Project Alpha?") == "TECHNICAL"


@requires_groq
def test_router_projects_question():
    assert classify_domain("What is Project Alpha?") == "PROJECTS"


@requires_groq
@requires_indexes
def test_workflow_end_to_end_hr_question():
    result = run_workflow("How many casual leaves do employees receive?")
    assert result["domain"] == "HR"
    assert "12" in result["answer"]
    assert len(result["citations"]) > 0


@requires_groq
@requires_indexes
def test_workflow_abstains_on_unknown_question():
    result = run_workflow("Who is the president of France?")
    assert NO_ANSWER_MESSAGE in result["answer"] or result["answer"] == NO_ANSWER_MESSAGE


def test_parse_follow_ups_handles_fenced_json():
    from src.agents import parse_follow_ups

    raw = '```json\n{"questions": ["A?", "B?", "C?", "D?"]}\n```'
    assert parse_follow_ups(raw) == ["A?", "B?", "C?"]


def test_parse_follow_ups_returns_empty_on_garbage():
    from src.agents import parse_follow_ups

    assert parse_follow_ups("not json at all") == []
