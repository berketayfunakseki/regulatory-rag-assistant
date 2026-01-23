# Regulatory RAG Assistant (FastAPI + Streamlit + FAISS)

A **portfolio-grade** project that demonstrates an end-to-end **Retrieval-Augmented Generation (RAG)** workflow:
document ingestion → chunking → semantic indexing (FAISS) → query search → API + dashboard.

This is designed to look **“Master application ready”** on GitHub.

## What you can do
- ✅ Ingest your own `.pdf` / `.txt` files
- ✅ Build a semantic index (Sentence Transformers + FAISS)
- ✅ Query via:
  - FastAPI endpoints (`/search`, `/ask`)
  - Streamlit UI (upload + search)
- ✅ LLM-free answer mode (safe default)  
  *(Optional: you can later plug in an LLM if you want.)*

## Project structure
```
regulatory-rag-assistant/
  src/
    ingest.py        # build the index
    rag.py           # vector search logic
    api.py           # FastAPI service
    app.py           # Streamlit dashboard
  data/
    docs/            # put your PDFs/TXTs here (gitignored)
    sample_docs/     # small sample files shipped with the repo
  index/             # FAISS index + metadata (gitignored)
  tests/
```
## Quickstart

### 1) Install
```bash
pip install -r requirements.txt
```

### 2) Build the index (using sample docs)
```bash
python -m src.ingest --use-samples
```

### 3) Run the API
```bash
uvicorn src.api:app --reload
```
Open: http://127.0.0.1:8000/docs

### 4) Run the dashboard
```bash
streamlit run -m src.app
```

## Example query
```bash
python -m src.query "What is the purpose of this policy?"
```

## Why this looks “elite” on GitHub
- Clean structure (src/data/index/tests)
- FastAPI + Streamlit (full-stack)
- FAISS semantic search (real ML component)
- Test suite ready (pytest)

## Next upgrades (optional)
- Add OpenAI / local LLM generation for final answers
- Add evaluation metrics (MRR/Recall@K)
- Deploy on Render / HuggingFace Spaces
