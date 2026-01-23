from __future__ import annotations

import argparse

from .config import DEFAULT_TOP_K
from .rag import VectorIndex, naive_answer_from_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RAG index.")
    parser.add_argument("question", type=str, help="Your question")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    idx = VectorIndex.load()
    hits = idx.search(args.question, top_k=args.top_k)
    print("\nTop matches:\n")
    for i, h in enumerate(hits, 1):
        print(f"{i}. score={h['score']:.3f} doc={h['doc']} chunk={h['chunk_id']}")
    print("\nAnswer (LLM-free):\n")
    print(naive_answer_from_chunks(args.question, hits))


if __name__ == "__main__":
    main()
