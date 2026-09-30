"""
Build one FAISS index per department from the PDFs under data/<department>/.

Usage:
    python scripts/build_indexes.py
"""

import sys
import pickle
from pathlib import Path

import faiss

# Allow running this script directly (python scripts/build_indexes.py)
# by adding the project root to sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, INDEX_DIR, DOMAIN_FOLDER
from src.ingestion import find_pdfs_by_department, process_pdf
from src.retriever import embed_texts


def build_index_for_department(folder_name: str, pdf_paths, department_label: str):
    print(f"\n=== Building index for department: {department_label} ===")

    all_chunks = []
    for pdf_path in pdf_paths:
        print(f"  Processing: {pdf_path.name}")
        raw_chunks = process_pdf(pdf_path, department=department_label)
        print(f"    -> {len(raw_chunks)} chunk(s) extracted")
        all_chunks.extend(raw_chunks)

    if not all_chunks:
        print(f"  No usable text found for '{department_label}'. Skipping index build.")
        return

    texts = [c.text for c in all_chunks]
    print(f"  Embedding {len(texts)} chunks...")
    embeddings = embed_texts(texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # inner product on normalized vectors ~= cosine
    index.add(embeddings)

    metadata = [
        {
            "chunk_text": c.text,
            "filename": c.filename,
            "page_number": c.page_number,
            "department": c.department,
        }
        for c in all_chunks
    ]

    out_dir = INDEX_DIR / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out_dir / "index.faiss"))
    with open(out_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"  Saved: {out_dir / 'index.faiss'}")
    print(f"  Saved: {out_dir / 'metadata.pkl'}")
    print(f"  Total vectors indexed: {index.ntotal}")


def main():
    print("AI Enterprise Knowledge Assistant — Index Builder")
    print(f"Data directory:  {DATA_DIR}")
    print(f"Index directory: {INDEX_DIR}")

    grouped_pdfs = find_pdfs_by_department(DATA_DIR)

    if not grouped_pdfs:
        print(
            "\nNo PDFs found under data/<department>/. "
            "Add PDFs to data/hr, data/technical, data/projects, "
            "or data/general and re-run this script."
        )
        return

    # department_label uses the canonical uppercase domain name for metadata,
    # while folder_name is used for the on-disk index path.
    folder_to_domain = {v: k for k, v in DOMAIN_FOLDER.items()}

    for folder_name, pdf_paths in grouped_pdfs.items():
        department_label = folder_to_domain.get(folder_name, folder_name.upper())
        build_index_for_department(folder_name, pdf_paths, department_label)

    print("\nAll indexes built successfully.")
    print("You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
