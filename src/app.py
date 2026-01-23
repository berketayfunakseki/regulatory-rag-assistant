from __future__ import annotations

import os
from pathlib import Path
import streamlit as st

from .config import DOCS_DIR, SAMPLES_DIR, DEFAULT_TOP_K
from .utils import list_docs
from .rag import VectorIndex, naive_answer_from_chunks
from .ingest import build_chunks
from .rag import Chunk


st.set_page_config(page_title="Regulatory RAG Assistant", page_icon="📚", layout="wide")


def ensure_dirs():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


def sidebar_docs():
    st.sidebar.header("📄 Documents")
    use_samples = st.sidebar.checkbox("Include sample docs", value=True)
    docs = list_docs(DOCS_DIR)
    if use_samples:
        docs += list_docs(SAMPLES_DIR)

    st.sidebar.caption(f"Indexed docs in data/docs ({len(list_docs(DOCS_DIR))} files)")
    if st.sidebar.button("Build / Rebuild Index"):
        paths = docs
        if not paths:
            st.sidebar.error("No documents found. Add .pdf/.txt to data/docs.")
        else:
            with st.spinner("Building index..."):
                chunks = []
                for p in paths:
                    # build_chunks expects list[Path] but we want per-file chunk ids stable
                    from .utils import load_document, chunk_text
                    raw = load_document(p)
                    parts = chunk_text(raw, chunk_chars=900, overlap=150)
                    for i, t in enumerate(parts):
                        chunks.append(Chunk(doc_name=p.name, chunk_id=i, text=t))
                idx = VectorIndex()
                idx.build(chunks)
                idx.save()
            st.sidebar.success("✅ Index built!")
    return docs


def uploader():
    st.sidebar.divider()
    st.sidebar.header("⬆️ Upload new docs"िलो)
    uploaded = st.sidebar.file_uploader("Upload .txt or .pdf", type=["txt", "pdf"], accept_multiple_files=True)
    if uploaded:
        ensure_dirs()
        for f in uploaded:
            out = DOCS_DIR / f.name
            out.write_bytes(f.getbuffer())
        st.sidebar.success(f"Saved {len(uploaded)} file(s) into data/docs")


def main():
    st.title("📚 Regulatory RAG Assistant")
    st.caption("Local RAG demo: ingest documents → semantic search → retrieve best passages." )

    docs = sidebar_docs()
    uploader()

    col1, col2 = st.columns([2, 1])
    with col1:
        query = st.text_input("Ask a question", placeholder="e.g., What are the main obligations for data processors?")
    with col2:
        top_k = st.number_input("Top K", min_value=1, max_value=20, value=DEFAULT_TOP_K, step=1)

    if st.button("Search"):
        try:
            idx = VectorIndex.load()
            hits = idx.search(query, top_k=int(top_k))
            st.subheader("Top matches")
            for h in hits:
                with st.expander(f"{h['doc']} — chunk {h['chunk_id']} (score {h['score']:.3f})"):
                    st.write(h["text"])
            st.subheader("Answer (LLM-free)")
            st.markdown(naive_answer_from_chunks(query, hits))
        except FileNotFoundError:
            st.error("Index not found. Click 'Build / Rebuild Index' in the sidebar.")
        except Exception as e:
            st.error(str(e))


if __name__ == "__main__":
    main()
