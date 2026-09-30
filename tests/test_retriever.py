"""
Tests for the retrieval layer.

These tests assume `python scripts/build_indexes.py` has already been run
using the sample PDFs (see scripts/generate_sample_pdfs.py).

Run with:
    pytest tests/test_retriever.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import retrieve, DomainIndexNotFound
from src.config import INDEX_DIR, DOMAIN_FOLDER


def _index_missing(domain: str) -> bool:
    folder = DOMAIN_FOLDER[domain]
    return not (INDEX_DIR / folder / "index.faiss").exists()


@pytest.mark.skipif(_index_missing("HR"), reason="HR index not built yet")
def test_hr_retrieval_returns_chunks():
    results = retrieve("How many casual leaves do employees receive?", "HR")
    assert len(results) > 0
    assert all(c.department == "HR" for c in results)


@pytest.mark.skipif(_index_missing("HR"), reason="HR index not built yet")
def test_hr_retrieval_preserves_metadata():
    results = retrieve("How many casual leaves do employees receive?", "HR")
    assert len(results) > 0
    top = results[0]
    assert top.filename.endswith(".pdf")
    assert isinstance(top.page_number, int)
    assert top.page_number > 0
    assert top.department == "HR"


@pytest.mark.skipif(_index_missing("TECHNICAL"), reason="TECHNICAL index not built yet")
def test_department_isolation_technical_does_not_return_hr_department_label():
    results = retrieve("How do I deploy Project Alpha?", "TECHNICAL")
    assert all(c.department == "TECHNICAL" for c in results)


def test_missing_index_raises_clear_error():
    with pytest.raises(DomainIndexNotFound):
        retrieve("does not matter", "NONEXISTENT_DOMAIN_XYZ")
