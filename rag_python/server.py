"""
FastAPI application entry point.

Run locally with:

    python server.py

Then open:

    http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile

from config import HOST, PORT
from rag.dedup.near import register_document
from rag.embeddings import embed_texts
from rag.generation.generation import generate_answer
from rag.ingestion.extractors import extract_text
from rag.ingestion.pipeline import ingest_document
from rag.retrieval.retrieval import retrieve
from rag.storage.db import (
    get_all_document_texts_for_near_dedup,
    get_all_indexable_chunks,
    init_db,
)
from rag.storage.indexing import vector_index
from schemas import ChatRequest, ChatResponse, SeedResponse, UploadResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    # -----------------------------------------------------------------------
    # Vector index
    # -----------------------------------------------------------------------
    #
    # Prefer a persisted FAISS index. If one does not exist, rebuild it from
    # the database and persist it for future startups.
    #
    # Qdrant is an external persistent service, so its own backend handles
    # persistence and no local FAISS rebuild is required.
    from config import FAISS_INDEX_PATH, VECTOR_BACKEND

    if VECTOR_BACKEND == "faiss":
        if vector_index.load_from_disk(FAISS_INDEX_PATH):
            print(
                "Loaded persisted vector index from disk "
                f"({vector_index.size} chunks)."
            )
        else:
            rows = get_all_indexable_chunks()

            if rows:
                contents = [row["embedding_text"] for row in rows]
                chunk_ids = [row["chunk_id"] for row in rows]

                vectors = embed_texts(contents)
                vector_index.add(vectors, chunk_ids)
                vector_index.save(FAISS_INDEX_PATH)

                print(
                    "No persisted vector index found -- rebuilt from DB "
                    f"({len(rows)} chunks) and saved it."
                )

    # -----------------------------------------------------------------------
    # Near-duplicate index
    # -----------------------------------------------------------------------
    #
    # Redis persists independently. The in-memory backend must be reconstructed
    # when the application starts.
    from config import NEAR_DUP_BACKEND

    if NEAR_DUP_BACKEND == "memory":
        doc_texts = get_all_document_texts_for_near_dedup()

        for doc_id, text in doc_texts.items():
            register_document(doc_id, text)

        if doc_texts:
            print(
                "Rebuilt near-duplicate index "
                f"({len(doc_texts)} documents)."
            )

    yield


app = FastAPI(
    title="Modular RAG Pipeline",
    description="A deliberately small RAG project, built up module by module.",
    version="0.3.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    text = extract_text(file.filename, raw)

    return ingest_document(file.filename, text)


@app.post("/seed", response_model=SeedResponse)
def seed():
    """Load the first 500 SQuAD passages into the pipeline."""
    from datasets import load_dataset

    ds = load_dataset("squad", split="train[:500]")

    breakdown: dict[str, int] = {}
    passages_seen = 0

    for i, row in enumerate(ds):
        context = row["context"]
        passages_seen += 1

        result = ingest_document(
            f"squad_passage_{i}.txt",
            context,
        )

        status = result["status"]
        breakdown[status] = breakdown.get(status, 0) + 1

    return {
        "status": "seeded",
        "passages_seen": passages_seen,
        "breakdown": breakdown,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    retrieved = retrieve(req.query)

    if not retrieved:
        return {
            "answer": (
                "No documents indexed yet -- "
                "try /seed or /upload first."
            ),
            "sources": [],
        }

    answer = generate_answer(req.query, retrieved)

    return {
        "answer": answer,
        "sources": [
            {"source": chunk["source"]}
            for chunk in retrieved
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)