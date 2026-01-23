from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import DEFAULT_TOP_K
from .rag import VectorIndex, naive_answer_from_chunks


app = FastAPI(title="Regulatory RAG Assistant", version="1.0.0")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20)


class AskResponse(BaseModel):
    query: str
    answer: str
    top_chunks: list


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search(req: SearchRequest):
    try:
        idx = VectorIndex.load()
        hits = idx.search(req.query, top_k=req.top_k)
        return {"query": req.query, "results": hits}
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Index not found. Run: python -m src.ingest --use-samples")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
def ask(req: SearchRequest):
    try:
        idx = VectorIndex.load()
        hits = idx.search(req.query, top_k=req.top_k)
        answer = naive_answer_from_chunks(req.query, hits)
        return AskResponse(query=req.query, answer=answer, top_chunks=hits)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Index not found. Run: python -m src.ingest --use-samples")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
