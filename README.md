# Modular RAG Pipeline

A deliberately small, modular RAG project, built up piece by piece to match
each concept as we cover it.

**Schema unchanged from last time -- if you already deleted rag.db and
haven't re-run seed yet, you're fine. If you have an old rag.db, delete it.**

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # optional
python server.py
```

Server runs at `http://127.0.0.1:8000`. Swagger UI: `http://127.0.0.1:8000/docs`.

## Project structure

```
config.py              <- every tunable value, used by every phase below
server.py               <- thin FastAPI wiring only
requirements.txt
rag/
├── storage/            <- the "document store" + "vector index" phase
│   ├── db.py
│   └── indexing.py
├── ingestion/           <- the "get documents in" phase
│   ├── extractors.py    (per-file-type text extraction: txt/md/html/pdf)
│   ├── chunking.py       (fixed or semantic chunking strategy)
│   ├── tables.py          (table -> separate embedding description)
│   └── pipeline.py        (orchestrates all of the above + dedup, per document)
├── dedup/                <- its own phase, three strategies side by side
│   ├── exact.py           (SHA-256 document/chunk hashing)
│   ├── near.py             (MinHash + LSH, document-level)
│   └── semantic.py          (embedding similarity, chunk-level)
├── embeddings.py           <- used by multiple phases, no folder needed yet
├── retrieval.py
└── generation.py
```

**The organizing principle**: `config.py` and `server.py` are the only two
files that sit at the root, because they're the only two things every phase
needs (config) or that wire every phase together (server). Everything else
lives under `rag/`, grouped into a folder once a phase has multiple related
files -- `storage/`, `ingestion/`, and `dedup/` each do; `embeddings.py`,
`retrieval.py`, `generation.py` stay flat since each is still one file.

Import style: every file imports `config` directly (e.g. `from config import
TOP_K`) since it's a project-root-level shared resource, not phase-specific.
Everything else is imported by its full phase path, e.g.
`from rag.storage.db import ...` or `from rag.dedup.near import ...` --
so the import line itself tells you which phase a piece of logic belongs to,
without needing to open the file.

## Where each concept lives in code

| Concept | File | Function |
|---|---|---|
| Fixed-size chunking w/ overlap | `rag/ingestion/chunking.py` | `_split_fixed()` |
| Semantic chunking | `rag/ingestion/chunking.py` | `_split_semantic()` |
| Small-to-retrieve, large-to-generate | `rag/ingestion/chunking.py`, `rag/storage/db.py` | `chunk_document()`, `get_chunks_with_parent_by_ids()` |
| Multi-doc-type extraction | `rag/ingestion/extractors.py` | `EXTRACTORS` registry |
| Table -> embedding description | `rag/ingestion/tables.py` | `get_embedding_text()` |
| Exact document/chunk duplicate (filename-independent) | `rag/storage/db.py` | `get_document_by_content_hash()` |
| Near-duplicate document (MinHash/LSH) | `rag/dedup/near.py` | `check_near_duplicate()` |
| Semantic chunk-level duplicate | `rag/dedup/semantic.py` | `find_semantic_duplicate()` |
| Full pipeline orchestration | `rag/ingestion/pipeline.py` | `ingest_document()` |
| Versioning + soft-delete | `rag/storage/db.py`, `rag/ingestion/pipeline.py` | `mark_document_stale()` |
| Vector removal (soft-delete's other half) | `rag/storage/indexing.py` | `VectorIndex.remove()` |
| ANN search (flat/exact for now) | `rag/storage/indexing.py` | `build_base_index()` |

## Try it

```bash
curl -X POST http://127.0.0.1:8000/seed
curl -X POST http://127.0.0.1:8000/chat \
    -H "Content-Type: application/json" -d '{"query": "What is the capital of France?"}'
curl -X POST http://127.0.0.1:8000/upload -F "file=@/path/to/some.pdf"
```

## Viewing the database

`rag.db` is a plain SQLite file -- VSCode "SQLite Viewer" extension, or
[DB Browser for SQLite](https://sqlitebrowser.org/), or `sqlite3 rag.db "..."`.

## Storage location -- what changing config.py actually buys you

`config.DB_PATH = "data/rag.db"` is the one line to edit if you want the
SQLite file somewhere else -- outside the repo entirely, a different local
path, whatever. Everything else imports `DB_PATH` from `config.py`, nothing
hardcodes it.

**Being precise about what this does and doesn't cover**: changing
`DB_PATH` only changes *which SQLite file* is used. Swapping to an entirely
different database *engine* (Postgres, for instance) would still require an
actual code change -- `rag/storage/db.py` uses `sqlite3.connect(...)`
directly, and a Postgres connection uses a different library (`psycopg2` or
similar) with different connection/query mechanics. The good news is that
change is **isolated to exactly one file** (`rag/storage/db.py`) thanks to
the phase-folder structure -- nothing in `ingestion/`, `dedup/`, or
`retrieval.py` would need to change, since they only ever call `db.py`'s
functions, never touch SQLite directly. So: config-only for "which SQLite
file," one-file-only (not zero-file) for "switch database engines entirely."

## Fixes applied during testing (kept here so future-you knows why)

- **Global exact-duplicate check was missing.** The original exact-dup
  check was scoped to `source_uri` (filename), so identical content under a
  *different* filename fell through to the more expensive near-dup (MinHash)
  path and still inserted a `documents` row -- costly at scale if a large
  fraction of ingested content is exact repeats. Fixed by adding
  `get_document_by_content_hash()` (a plain indexed lookup, filename-
  independent) as step 0 in `ingest_document()`, before versioning or
  near-dup logic even run.
- **Near-dup tests used unrealistically short text.** MinHash near-dup
  detection uses 9-word shingles -- meaningful for paragraph-length content,
  but a 14-word test sentence with one word changed only reaches ~0.77
  similarity against the 0.8 threshold (verified empirically). Tests now use
  paragraph-length text, matching what the algorithm is actually designed
  for and what real documents look like.
- **FAISS + PyTorch OpenMP segfault** -- see the dedicated section below.



- Indexing: `IndexFlatIP` does exact brute-force search. HNSW/IVF next --
  this will be a contained change inside `rag/storage/indexing.py` only.
- Retrieval: dense search only -- no BM25/hybrid, no reranking, no query rewriting yet.
- Table handling detects already-Markdown-formatted tables within a chunk;
  no multi-page table reconstruction.
- Versioning uses filename as the identity key across versions.
- `rag/dedup/near.py`'s LSH index is in-memory only, rebuilt each restart.

## Running tests

```bash
pip install -r requirements.txt   # now includes pytest + httpx
pytest tests/ -v
```

Tests are structured to mirror the pipeline itself:
- `test_dedup_exact.py`, `test_dedup_near.py` -- each dedup strategy tested in isolation
- `test_chunking.py`, `test_tables.py`, `test_extractors.py` -- ingestion-phase logic, pure functions, no DB/network needed
- `test_storage_db.py`, `test_storage_indexing.py` -- the two storage systems, including versioning/soft-delete and vector removal
- `test_ingestion_pipeline.py` -- integration tests hitting `ingest_document()` end-to-end (exact/near dup, versioning)
- `test_retrieval.py` -- retrieval mechanics (embed -> search -> fetch parent)
- `test_api.py` -- real HTTP requests against the FastAPI app, including startup/shutdown lifespan

**Fake embeddings, not a real model**: most tests use a `fake_embeddings`
fixture (see `conftest.py`) -- deterministic, instant vectors instead of
loading `sentence-transformers`, so the suite runs in seconds. They're NOT
semantically meaningful (don't test "does retrieval find the most relevant
result among several options" with them) -- only real-model tests could
verify that; this suite verifies the *plumbing*, not embedding quality.

**Every test gets a fresh DB, vector index, and MinHash/LSH index**
automatically (autouse fixtures in `conftest.py`) -- no manual cleanup, and
no state leaks between tests.

## Known issue: FAISS + PyTorch segfault on macOS

If you see `Fatal Python error: Segmentation fault` inside
`faiss/swigfaiss.py`'s `search()`, this is a known compatibility problem,
not a bug in the pipeline logic: both `faiss-cpu` and `torch` (pulled in by
`sentence-transformers`) bundle their own OpenMP runtime, and loading both
in one process can crash when FAISS spins up its parallel search threads.

**Already fixed** in `server.py` and `tests/conftest.py` (env vars set at
the very top, before any other import) and in `rag/storage/indexing.py`
(FAISS capped to a single thread). If you still hit it:
- Make sure you're running the latest version of these three files.
- As a last resort, set the env vars manually before running anything:
  `export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1`
- The more "correct" long-term fix (not needed here, just FYI) is installing
  `faiss-cpu` from `conda-forge` instead of the pip wheel, since conda
  coordinates shared native libraries across packages instead of each
  bundling its own copy.