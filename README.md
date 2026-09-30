# AI Enterprise Knowledge Assistant — MVP

A minimal, locally runnable Retrieval-Augmented Generation (RAG) system that
answers employee questions using only the company's own PDF documents,
routed to the correct department before retrieval.

## Project Overview

Employees ask natural-language questions ("How many casual leaves do I
get?", "How do I deploy Project Alpha?"). The system:

1. Classifies the question into a department (HR / TECHNICAL / PROJECTS / GENERAL).
2. Searches only that department's FAISS vector index for relevant chunks.
3. Asks an LLM (Groq's `llama-3.3-70b-versatile`) to answer **strictly**
   from those chunks.
4. Shows the answer with citations (filename + page number) built from
   retrieval metadata — never invented by the LLM.
5. If nothing relevant is found, it abstains instead of guessing.

## Architecture

```
                  ┌─────────────────┐
                  │   Streamlit UI  │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Manager Agent  │
                  │ Domain Routing  │
                  └────────┬────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
             HR       TECHNICAL      PROJECTS
              │            │            │
              └────────────┼────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ FAISS Retriever │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Retrieved Chunks│
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Answer Agent   │
                  │   Groq LLM      │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Answer + Source │
                  └─────────────────┘
```

There are exactly **two LLM-oriented responsibilities**: the Manager Agent
(routing only) and the Answer Agent (grounded generation only). Retrieval is
plain Python — no LLM involved.

## Technology Stack

| Layer               | Choice                          |
|---------------------|----------------------------------|
| UI                   | Streamlit                       |
| Orchestration        | LangGraph                       |
| LLM                  | Groq — `llama-3.3-70b-versatile`|
| Embeddings           | Sentence Transformers — `all-MiniLM-L6-v2` |
| Vector search         | FAISS (`IndexFlatIP`, per department) |
| PDF extraction        | PyMuPDF                         |
| Config                | python-dotenv                   |
| Data validation        | Pydantic                        |

## Folder Structure

```
enterprise-rag-mvp/
│
├── app.py                      # Streamlit chat UI
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── data/                       # Source PDFs, grouped by department
│   ├── hr/
│   ├── technical/
│   ├── projects/
│   └── general/
│
├── indexes/                    # FAISS index + metadata per department
│   ├── hr/
│   ├── technical/
│   ├── projects/
│   └── general/
│
├── scripts/
│   ├── build_indexes.py        # Ingestion pipeline
│   └── generate_sample_pdfs.py # Creates the 4 sample PDFs
│
├── src/
│   ├── config.py                # All env-driven settings
│   ├── schemas.py                # Pydantic models (Chunk, Citation, RouterOutput)
│   ├── ingestion.py              # PDF extraction + chunking
│   ├── retriever.py              # Embedding + FAISS search (plain Python)
│   ├── agents.py                 # Manager Agent + Answer Agent (Groq)
│   └── workflow.py               # LangGraph: Manager -> Retrieval -> Answer
│
└── tests/
    ├── test_retriever.py
    └── test_workflow.py
```

## Installation

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Environment Setup

Copy the example env file and add your Groq API key
(get one free at https://console.groq.com):

```bash
cp .env.example .env
```

```text
GROQ_API_KEY=your_api_key
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K=5
SIMILARITY_THRESHOLD=0.35
CHUNK_SIZE=800
CHUNK_OVERLAP=100
```

`SIMILARITY_THRESHOLD=0.35` is a **starting point, not a universal
truth** — cosine-similarity thresholds depend on the embedding model and
your document vocabulary. Tune it against your own real documents and
questions.

## Adding PDFs

Drop PDFs into the matching department folder:

```
data/hr/*.pdf
data/technical/*.pdf
data/projects/*.pdf
data/general/*.pdf
```

The department is inferred from the parent folder name.

The repo already ships four sample PDFs (see "Sample Documents" below). To
regenerate them from scratch:

```bash
python scripts/generate_sample_pdfs.py
```

## Building FAISS Indexes

```bash
python scripts/build_indexes.py
```

This will, for every department folder that contains at least one PDF:
extract text page-by-page → chunk it (800 chars, 100 overlap by default) →
embed with `all-MiniLM-L6-v2` → build a normalized `IndexFlatIP` FAISS
index → save `indexes/<department>/index.faiss` and
`indexes/<department>/metadata.pkl` (filename, page number, department per
chunk).

## Running Streamlit

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

## Example Questions

| Question | Expected Domain | Expected behavior |
|---|---|---|
| How many casual leaves do employees receive? | HR | "12 casual leaves..." + `leave_policy.pdf — Page 4` |
| How do I deploy Project Alpha? | TECHNICAL | Deployment steps from `deployment_manual.pdf` |
| What is Project Alpha? | PROJECTS | Overview from `project_alpha.pdf` |
| What are the company working hours? | GENERAL | "9:00 AM to 6:00 PM" from `company_information.pdf` |
| Who is the president of France? | any | "I couldn't find enough information in the company documents to answer this question." |

## How RAG Works (for the viva)

RAG = **R**etrieval-**A**ugmented **G**eneration. Instead of asking the LLM
to answer from its own trained knowledge (which can be outdated or
hallucinated), we:

1. Convert the question and all document chunks into numeric vectors
   (embeddings) that capture meaning, not just keywords.
2. Find the chunks whose vectors are closest to the question's vector
   (cosine similarity via FAISS inner product on normalized vectors).
3. Paste those chunks into the LLM's prompt as context and instruct it to
   answer *only* from that context.

This keeps answers grounded in real company documents and makes them
auditable via citations, instead of relying on the model's possibly stale
or incorrect internal knowledge.

## How LangGraph Works (for the viva)

LangGraph models the pipeline as an explicit state machine instead of one
big function. Each node reads and updates a shared `GraphState`
(`question`, `domain`, `retrieved_chunks`, `answer`, `citations`):

```
START → manager_node → retrieval_node → answer_node → END
```

- `manager_node` calls the Manager Agent to set `state["domain"]`.
- `retrieval_node` is plain Python: calls `retrieve()` and sets
  `state["retrieved_chunks"]`.
- `answer_node` calls the Answer Agent and builds `state["citations"]`
  from retrieval metadata.

This makes the flow easy to trace, test node-by-node, and extend later
(e.g. adding a query-rewriting node) without touching the rest of the
graph.

## Known Limitations

This is an MVP. The following are intentionally **not** implemented, and
are listed here as future improvements rather than gaps in scope:

- No authentication or user management
- No database (PostgreSQL/Redis) — FAISS + pickle files only
- No Docker/Kubernetes/cloud deployment
- No OCR (scanned/image-only PDFs will yield no text)
- No hybrid search (BM25 + embeddings) or cross-encoder reranking
- No query rewriting or multi-query retrieval
- No conversation memory (each question is answered independently)
- No background task queue (Celery)
- No advanced observability/evaluation dashboards
- Single LLM provider (Groq) — no fallback provider
- No fine-tuning
- The similarity threshold and chunk size are simple defaults that should
  be tuned against real company documents before production use

## Testing

```bash
pytest tests/ -v
```

Tests that require FAISS indexes or a configured `GROQ_API_KEY` are
automatically skipped if those prerequisites aren't available, so the test
suite is safe to run at any stage of setup.
