"""
server.py -- thin FastAPI wiring only. All logic lives in the other files.

Run:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...     # optional
    python server.py

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""

# MUST run before faiss or torch/sentence-transformers get imported anywhere
# (directly or transitively below) -- both bundle their own OpenMP runtime,
# and loading both in one process can otherwise segfault inside FAISS's
# search(). See README for the full explanation.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from config import HOST, PORT
from rag.storage.db import init_db, get_all_indexable_chunks
from rag.embeddings import embed_texts
from rag.storage.indexing import vector_index
from rag.ingestion.pipeline import ingest_document
from rag.ingestion.extractors import extract_text
from rag.retrieval import retrieve
from rag.generation import generate_answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    rows = get_all_indexable_chunks()
    if rows:
        contents = [r["embedding_text"] for r in rows]
        chunk_ids = [r["chunk_id"] for r in rows]
        vectors = embed_texts(contents)
        vector_index.add(vectors, chunk_ids)
        print(f"Rebuilt vector index with {len(rows)} indexable chunks from existing DB.")
    yield


app = FastAPI(
    title="Modular RAG Pipeline",
    description="A deliberately small RAG project, built up module by module.",
    version="0.3.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------
class UploadResponse(BaseModel):
    status: str
    doc_id: str
    version: Optional[int] = None
    chunks: Optional[int] = None
    duplicate_chunks_skipped: Optional[int] = None
    near_dup_of: Optional[str] = None


class ChatRequest(BaseModel):
    query: str


class SourceInfo(BaseModel):
    source: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


class SeedResponse(BaseModel):
    status: str
    passages_seen: int
    breakdown: dict[str, int]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    text = extract_text(file.filename, raw)   # dispatches by extension: txt/md/html/pdf
    return ingest_document(file.filename, text)


@app.post("/seed", response_model=SeedResponse)
def seed():
    """Pull passages from SQuAD. Deliberately does NOT pre-filter duplicate
    contexts -- SQuAD naturally repeats the same context across questions,
    exercising the exact/near/semantic dedup logic for real."""
    from datasets import load_dataset

    ds = load_dataset("squad", split="train[:500]")
    breakdown: dict[str, int] = {}
    passages_seen = 0

    for i, row in enumerate(ds):
        ctx = row["context"]
        passages_seen += 1
        result = ingest_document(f"squad_passage_{i}.txt", ctx)
        breakdown[result["status"]] = breakdown.get(result["status"], 0) + 1

    return {"status": "seeded", "passages_seen": passages_seen, "breakdown": breakdown}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    retrieved = retrieve(req.query)
    if not retrieved:
        return {"answer": "No documents indexed yet -- try /seed or /upload first.", "sources": []}

    answer = generate_answer(req.query, retrieved)
    return {
        "answer": answer,
        "sources": [{"source": c["source"], "score": c["score"]} for c in retrieved],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)