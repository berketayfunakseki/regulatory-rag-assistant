from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
SAMPLES_DIR = DATA_DIR / "sample_docs"

INDEX_DIR = ROOT / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
STORE_PATH = INDEX_DIR / "store.json"

DEFAULT_CHUNK_CHARS = 900
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
