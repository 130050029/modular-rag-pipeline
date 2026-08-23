"""
server.py -- thin FastAPI wiring only. All logic lives in the other files.

Run:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...     # optional
    python server.py

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""

# The FAISS + torch OpenMP workaround (KMP_DUPLICATE_LIB_OK/OMP_NUM_THREADS)
# is now centralized in config.py, imported first below -- see its comment
# for why it lives there rather than being repeated in every entry point.


from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File

from config import HOST, PORT
from rag.storage.db import init_db, get_all_indexable_chunks, get_all_document_texts_for_near_dedup
from rag.embeddings import embed_texts
from rag.storage.indexing import vector_index
from rag.dedup.near import register_document
from rag.ingestion.pipeline import ingest_document
from rag.ingestion.extractors import extract_text
from rag.retrieval.retrieval import retrieve
from rag.generation.generation import generate_answer

from schemas import SeedResponse, UploadResponse, ChatRequest, ChatResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    # --- Vector index: try loading a persisted one first ----------------------
    # Without this, every restart re-embeds the ENTIRE corpus from scratch --
    # fine at toy scale, a genuinely severe problem at millions of chunks
    # (a restart could take hours instead of being instant). Only fall back
    # to the expensive full rebuild if nothing was persisted yet (e.g. first
    # run, or the index files were deleted).
    from config import FAISS_INDEX_PATH   # read fresh here, not at module top,
                                            # so a monkeypatched value (tests) is honored
    if vector_index.load_from_disk(FAISS_INDEX_PATH):
        print(f"Loaded persisted vector index from disk ({vector_index.size} chunks) -- skipped re-embedding.")
    else:
        rows = get_all_indexable_chunks()
        if rows:
            contents = [r["embedding_text"] for r in rows]
            chunk_ids = [r["chunk_id"] for r in rows]
            vectors = embed_texts(contents)
            vector_index.add(vectors, chunk_ids)
            vector_index.save(FAISS_INDEX_PATH)
            print(f"No persisted index found -- rebuilt from DB ({len(rows)} chunks) and saved for next startup.")

    # --- Near-duplicate index: only the "memory" backend needs rebuilding -----
    # The "redis" backend persists itself (as long as Redis's own data
    # directory is bind-mounted -- see docker-compose.yml), so nothing to do
    # there. The in-memory backend has no persistence of its own at all, so
    # every restart otherwise loses near-dup awareness of already-ingested
    # documents entirely. We don't store a document's raw original text
    # anywhere, so this reconstructs it from parent chunks (see
    # get_all_document_texts_for_near_dedup's docstring for the caveat on
    # exactness) -- cheap relative to re-embedding, since MinHash is just
    # hashing, no model inference involved.
    from config import NEAR_DUP_BACKEND   # read fresh here, not at module top,
                                            # so a monkeypatched value (tests) is honored
    if NEAR_DUP_BACKEND == "memory":
        doc_texts = get_all_document_texts_for_near_dedup()
        for doc_id, text in doc_texts.items():
            register_document(doc_id, text)
        if doc_texts:
            print(f"Rebuilt near-duplicate index (memory backend) with {len(doc_texts)} documents.")

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
        "sources": [{"source": c["source"]} for c in retrieved],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)