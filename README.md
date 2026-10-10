# Regulatory RAG Assistant

[![CI](https://github.com/berketayfunakseki/regulatory-rag-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/berketayfunakseki/regulatory-rag-assistant/actions/workflows/ci.yml)

An end-to-end Retrieval-Augmented Generation system for querying regulatory documents. Ingests PDFs and text files, builds a semantic index, and lets you search or ask questions through a REST API or a Streamlit UI.

## Pipeline

```
PDF / TXT documents
      │  chunking
      v
Sentence-Transformer embeddings
      │
      v
FAISS vector index  ──▶  top-k semantic retrieval
      │
      v
FastAPI /search, /ask  ◀──▶  Streamlit UI
```

CI runs lint + tests on every push via GitHub Actions.

## Features

- Ingests your own `.pdf` / `.txt` files, or use the bundled sample docs
- Builds a FAISS semantic index from Sentence-Transformer embeddings
- Query via FastAPI (`/search`, `/ask`) or the Streamlit dashboard
- LLM-free retrieval mode by default — no API key required to run it

## Project structure

```
regulatory-rag-assistant/
  src/
    ingest.py        # build the index
    rag.py            # vector search logic
    api.py            # FastAPI service
    app.py            # Streamlit dashboard
  data/
    docs/             # your own PDFs/TXTs (gitignored)
    sample_docs/      # sample files shipped with the repo
  index/              # FAISS index + metadata (gitignored)
  tests/
```

## Running locally

```bash
pip install -r requirements.txt

# build the index using the bundled sample docs
python -m src.ingest --use-samples

# run the API
uvicorn src.api:app --reload
# → http://127.0.0.1:8000/docs

# or run the dashboard
streamlit run src/app.py
```

## Example query

```bash
python -m src.query "What is the purpose of this policy?"
```

## Why FAISS

An in-memory vector index gives millisecond retrieval with no external vector-DB service to run or pay for — appropriate for a document set of this size.

## What I'd improve next

- Plug in an LLM (OpenAI or local) for generative answers instead of raw retrieval
- Add retrieval evaluation metrics (Recall@K, MRR)
- Deploy to Render or Hugging Face Spaces for a live demo

---
Berke Tayfun Akseki — [berketayfunakseki.com](https://berketayfunakseki.com)
