"""
Retrieval layer: embeds a query with Sentence Transformers and searches the
FAISS index for a single department (never across all departments).

This module is plain Python — it is intentionally NOT an LLM agent.
"""

import pickle
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, TOP_K, SIMILARITY_THRESHOLD, index_paths
from src.schemas import Chunk

_model = None  # lazy-loaded singleton so we only load the model once


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of texts and L2-normalize so inner product ~ cosine similarity."""
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
    return embeddings


class DomainIndexNotFound(Exception):
    pass


def load_domain_index(domain: str):
    """Load the FAISS index + metadata for a single domain."""
    index_path, metadata_path = index_paths(domain)

    if not index_path.exists() or not metadata_path.exists():
        raise DomainIndexNotFound(
            f"No index found for domain '{domain}'. "
            f"Run: python scripts/build_indexes.py"
        )

    index = faiss.read_index(str(index_path))
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata


def retrieve(query: str, domain: str, top_k: int = TOP_K) -> List[Chunk]:
    """
    Embed `query`, search only the FAISS index belonging to `domain`,
    and return the top_k chunks (with similarity scores and metadata)
    that pass the similarity threshold.
    """
    index, metadata = load_domain_index(domain)

    if index.ntotal == 0:
        return []

    query_vec = embed_texts([query])
    scores, indices = index.search(query_vec, min(top_k, index.ntotal))

    results: List[Chunk] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        if score < SIMILARITY_THRESHOLD:
            continue
        meta = metadata[idx]
        results.append(
            Chunk(
                text=meta["chunk_text"],
                filename=meta["filename"],
                page_number=meta["page_number"],
                department=meta["department"],
                score=float(score),
            )
        )

    return results
