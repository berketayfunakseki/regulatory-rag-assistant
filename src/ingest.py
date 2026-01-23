from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .config import DOCS_DIR, SAMPLES_DIR, DEFAULT_CHUNK_CHARS, DEFAULT_CHUNK_OVERLAP
from .utils import load_document, chunk_text, list_docs
from .rag import VectorIndex, Chunk


def collect_paths(use_samples: bool = False) -> List[Path]:
    paths = list_docs(DOCS_DIR)
    if use_samples:
        paths += list_docs(SAMPLES_DIR)
    return paths


def build_chunks(paths: List[Path], chunk_chars: int, overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for p in paths:
        raw = load_document(p)
        parts = chunk_text(raw, chunk_chars=chunk_chars, overlap=overlap)
        for i, t in enumerate(parts):
            chunks.append(Chunk(doc_name=p.name, chunk_id=i, text=t))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents and build a FAISS index.")
    parser.add_argument("--use-samples", action="store_true", help="Include sample docs shipped with the repo.")
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    args = parser.parse_args()

    paths = collect_paths(use_samples=args.use_samples)
    if not paths:
        print("No documents found. Put .pdf/.txt into data/docs or run with --use-samples.")
        return

    chunks = build_chunks(paths, chunk_chars=args.chunk_chars, overlap=args.overlap)
    if not chunks:
        print("No text extracted from the documents.")
        return

    idx = VectorIndex()
    idx.build(chunks)
    idx.save()
    print(f"✅ Built index with {len(chunks)} chunks from {len(paths)} documents.")


if __name__ == "__main__":
    main()
