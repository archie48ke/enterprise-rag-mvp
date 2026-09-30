"""
PDF ingestion utilities.

Responsible for:
1. Extracting text from PDFs page-by-page using PyMuPDF.
2. Splitting page text into overlapping character chunks.
3. Preserving filename / page_number / department metadata for every chunk.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pymupdf as fitz  # PyMuPDF (current import name; `fitz` alias is deprecated)

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class RawChunk:
    text: str
    filename: str
    page_number: int
    department: str


def extract_pages(pdf_path: Path) -> List[str]:
    """
    Extract text from a PDF, one string per page.

    Returns an empty list (and lets the caller decide to skip the file) if
    the PDF is empty or cannot be opened.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:  # corrupted / unreadable PDF
        print(f"[ingestion] Could not open {pdf_path.name}: {exc}")
        return []

    pages = []
    try:
        for page in doc:
            pages.append(page.get_text("text"))
    except Exception as exc:
        print(f"[ingestion] Error reading pages from {pdf_path.name}: {exc}")
        return []
    finally:
        doc.close()

    return pages


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Simple sliding-window character chunker.

    Not sentence- or token-aware on purpose: it is easy to explain and
    reason about for an MVP, and works well enough for short policy /
    technical documents.
    """
    text = text.strip()
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == text_len:
            break
        start = end - chunk_overlap

    return chunks


def process_pdf(pdf_path: Path, department: str) -> List[RawChunk]:
    """
    Full pipeline for a single PDF: extract -> chunk -> attach metadata.
    Skips empty PDFs. Never raises on a corrupted PDF (returns []).
    """
    pages = extract_pages(pdf_path)

    if not pages or all(not p.strip() for p in pages):
        print(f"[ingestion] Skipping empty PDF: {pdf_path.name}")
        return []

    raw_chunks: List[RawChunk] = []
    for page_number, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        for chunk in chunk_text(page_text):
            raw_chunks.append(
                RawChunk(
                    text=chunk,
                    filename=pdf_path.name,
                    page_number=page_number,
                    department=department,
                )
            )

    return raw_chunks


def find_pdfs_by_department(data_dir: Path) -> dict:
    """
    Walk data/<department>/*.pdf and group PDF paths by department
    (department name = parent folder name, lower-cased).
    """
    grouped: dict = {}
    if not data_dir.exists():
        return grouped

    for dept_dir in sorted(data_dir.iterdir()):
        if not dept_dir.is_dir():
            continue
        pdfs = sorted(dept_dir.glob("*.pdf"))
        if pdfs:
            grouped[dept_dir.name.lower()] = pdfs

    return grouped
