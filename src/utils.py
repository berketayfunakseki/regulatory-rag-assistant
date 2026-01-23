from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

import pdfplumber


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path: Path) -> str:
    text_parts: List[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                text_parts.append(txt)
    return "\n\n".join(text_parts)


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return read_text_file(path)
    if suffix == ".pdf":
        return read_pdf_file(path)
    raise ValueError(f"Unsupported file type: {path.name}")


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


def list_docs(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".txt", ".pdf"}])
