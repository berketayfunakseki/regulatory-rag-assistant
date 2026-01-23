from pathlib import Path
from src.utils import chunk_text
from src.rag import Chunk, VectorIndex


def test_chunking():
    text = "A" * 2000
    chunks = chunk_text(text, chunk_chars=900, overlap=150)
    assert len(chunks) >= 2


def test_index_build_and_search(tmp_path: Path):
    chunks = [
        Chunk(doc_name="a.txt", chunk_id=0, text="Data protection is important."),
        Chunk(doc_name="b.txt", chunk_id=0, text="Marketing analytics can predict churn."),
    ]
    idx = VectorIndex()
    idx.build(chunks)
    hits = idx.search("data protection", top_k=2)
    assert len(hits) == 2
