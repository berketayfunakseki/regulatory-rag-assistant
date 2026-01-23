from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    faiss = None  # type: ignore

from sentence_transformers import SentenceTransformer  # type: ignore

from .config import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_TOP_K,
    FAISS_INDEX_PATH,
    STORE_PATH,
)


@dataclass
class Chunk:
    doc_name: str
    chunk_id: int
    text: str


class VectorIndex:
    """FAISS cosine-similarity index (normalized inner product)."""

    def __init__(self, embed_model: str = DEFAULT_EMBED_MODEL):
        if faiss is None:  # pragma: no cover
            raise RuntimeError("faiss is not available. Install faiss-cpu.")
        self.embed_model_name = embed_model
        self.model = SentenceTransformer(embed_model)
        self.index = None
        self.chunks: List[Chunk] = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32)

    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]
        vecs = self._embed(texts)
        dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vecs)

    def save(self, index_path: Path = FAISS_INDEX_PATH, store_path: Path = STORE_PATH) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.parent.mkdir(parents=True, exist_ok=True)

        if self.index is None:
            raise RuntimeError("Index not built")

        faiss.write_index(self.index, str(index_path))
        store = {
            "embed_model": self.embed_model_name,
            "chunks": [c.__dict__ for c in self.chunks],
        }
        store_path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_path: Path = FAISS_INDEX_PATH, store_path: Path = STORE_PATH) -> "VectorIndex":
        if faiss is None:  # pragma: no cover
            raise RuntimeError("faiss is not available. Install faiss-cpu.")

        store = json.loads(store_path.read_text(encoding="utf-8"))
        obj = cls(embed_model=store.get("embed_model", DEFAULT_EMBED_MODEL))
        obj.chunks = [Chunk(**c) for c in store["chunks"]]
        obj.index = faiss.read_index(str(index_path))
        return obj

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Index not built / loaded")
        qv = self._embed([query])
        scores, ids = self.index.search(qv, top_k)
        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
            if idx < 0 or idx >= len(self.chunks):
                continue
            c = self.chunks[idx]
            results.append({
                "score": float(score),
                "doc": c.doc_name,
                "chunk_id": c.chunk_id,
                "text": c.text,
            })
        return results


def naive_answer_from_chunks(question: str, chunks: List[Dict[str, Any]]) -> str:
    """LLM-free answer: return the best chunk plus a short rationale."""
    if not chunks:
        return "I couldn't find relevant passages in the indexed documents."
    best = chunks[0]
    return (
        f"Most relevant passage appears in **{best['doc']}** (chunk {best['chunk_id']}).\n\n"
        f"{best['text']}"
    )
