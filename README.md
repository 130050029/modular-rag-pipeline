# Modular RAG Pipeline

<!-- Replace YOUR_USERNAME below with your actual GitHub username once pushed -->
![Tests](https://github.com/130050029/modular-rag-pipeline/actions/workflows/tests.yml/badge.svg)

A modular, end-to-end Retrieval-Augmented Generation (RAG) pipeline built with FastAPI, FAISS, and sentence-transformers. It implements document ingestion with multi-format extraction, a three-layer deduplication strategy, document versioning with soft-delete, parent/child chunking, and table-aware embedding — each concern isolated into its own module so individual components can be understood, tested, and replaced independently.

## Features

- **Multi-format ingestion**: plain text, Markdown, HTML, PDF, and (via Docling) images, dispatched through a pluggable extractor registry with a configurable PDF/image extraction method (PyMuPDF or Docling)
- **Three-layer deduplication**: exact (content hash), near-duplicate (MinHash + LSH), and semantic (embedding similarity)
- **Document versioning**: re-ingesting an updated document soft-deletes the previous version and removes its vectors from the live index
- **Parent/child chunking**: small chunks are embedded and searched; their parent chunks are what get passed to the LLM, keeping retrieval precise and generated context complete
- **Two configurable chunking strategies**: fixed-size word windows, or semantic boundary detection based on sentence-embedding similarity
- **Table-aware embedding**: tables are detected and embedded via a generated natural-language description rather than their raw structure
- **FAISS-backed vector search** with support for removal (required for soft-delete)
- **Full test suite** (pytest) covering every module in isolation plus end-to-end integration tests

## Architecture

```
                    ┌─────────────────────────────┐
                    │  Client (Swagger UI / curl)  │
                    └───────────────┬─────────────┘
                                    │
                    ┌───────────────▼─────────────┐
                    │  server.py -- FastAPI (thin) │
                    │  lifespan(): load/rebuild    │
                    │  indexes on startup          │
                    └──┬────────────────────────┬──┘
                       │                        │
        ┌──────────────▼─────────────┐   ┌──────▼───────────────────────┐
        │  INGESTION                 │   │  RETRIEVAL + GENERATION      │
        │  extractors.py             │   │  retrieval.py                │
        │  chunking.py               │   │    embed query -> search ->  │
        │  dedup/ (exact->near->sem) │   │    fetch parent content      │
        │  pipeline.py orchestrates  │   │  generation.py                │
        │    all of the above        │   │    prompt + Claude API call   │
        └──────────────┬─────────────┘   └───────────────┬───────────────┘
                       │                                  │
                       └───────────────┬──────────────────┘
                                       │
                         ┌─────────────▼─────────────┐
                         │  STORAGE                  │
                         │  db.py + connection.py     │
                         │    (SQLite or Postgres)     │
                         │  indexing.py                 │
                         │    (FAISS HNSW/flat, or       │
                         │     Qdrant service)             │
                         └───────┬───────┬───────┬────────┘
                                 │       │       │
                    ┌────────────┘       │       └─────────────┐
                    ▼                    ▼                     ▼
              ┌──────────┐        ┌──────────┐          ┌──────────────┐
              │ Postgres │        │  Redis   │          │   Qdrant     │
              │(optional)│        │(optional)│          │  (optional)  │
              └──────────┘        └──────────┘          └──────────────┘

        Also optional, wired in separately: Docling (extraction) and the
        Anthropic API (generation) -- see docker-compose.yml and config.py.
```

```
config.py              Every tunable value in the project; nothing else hardcodes settings
server.py               FastAPI application (thin wiring layer only)
requirements.txt
rag/
├── storage/            Document store (SQLite or Postgres) and vector index (FAISS)
│   ├── db.py
│   ├── connection.py     Backend-agnostic connection layer (SQLite/Postgres)
│   └── indexing.py
├── ingestion/           Document intake: extraction, chunking, table handling, orchestration
│   ├── extractors.py    Per-file-type text extraction (txt / md / html / pdf)
│   ├── chunking.py       Fixed-size or semantic chunking strategy
│   ├── tables.py          Table detection and embedding-description generation
│   └── pipeline.py        Orchestrates extraction, dedup, chunking, and indexing per document
├── dedup/                Three deduplication strategies
│   ├── exact.py           SHA-256 content hashing
│   ├── near.py             MinHash + LSH (document-level)
│   └── semantic.py          Embedding similarity (chunk-level)
├── embeddings.py           Embedding model wrapper
├── retrieval.py
└── generation.py
data/
├── manual_test_files/     Sample files for manual endpoint testing
└── golden_set/             Query set for retrieval evaluation (Recall@k / MRR)
tests/                     pytest suite, mirroring the structure above
```

**Design principle**: `config.py` and `server.py` are the only files at the project root — they are the two things every module depends on (configuration) or that tie every module together (the API layer). Every other concern lives under `rag/`, grouped into a subpackage once it has multiple related files (`storage/`, `ingestion/`, `dedup/`); single-file concerns (`embeddings.py`, `retrieval.py`, `generation.py`) remain flat.

Import convention: `config` is imported directly everywhere (`from config import TOP_K`) since it is a shared, project-wide resource. Everything else is imported by its full package path (`from rag.storage.db import ...`, `from rag.dedup.near import ...`), so the import statement itself identifies which layer of the pipeline a piece of logic belongs to.

## Installation

```bash
git clone <this-repo>
cd modular-rag-pipeline
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # optional; without it, /chat returns raw retrieved context instead of a generated answer
```

## Running

```bash
python server.py
```

The API is available at `http://127.0.0.1:8000`, with interactive documentation (Swagger UI) at `http://127.0.0.1:8000/docs`.

## Usage

```bash
# Ingest ~500 sample passages from the SQuAD dataset
curl -X POST http://127.0.0.1:8000/seed

# Ask a question
curl -X POST http://127.0.0.1:8000/chat \
    -H "Content-Type: application/json" -d '{"query": "What is the capital of France?"}'

# Upload a document (txt, md, html, or pdf)
curl -X POST http://127.0.0.1:8000/upload -F "file=@/path/to/document.pdf"
```

See `data/README.md` for a curated set of sample files that exercise every ingestion behavior (multi-format extraction, near-duplicate detection, table handling, versioning), and `data/golden_set/README.md` for the retrieval evaluation query set.

## Configuration

All configurable values live in `config.py`, including the storage backend, embedding model, chunking strategy and sizes, and deduplication thresholds. No other file hardcodes these values.

### Storage backend: SQLite or Postgres

Set via `config.DB_BACKEND` (or the `DB_BACKEND` environment variable): `"sqlite"` (default) or `"postgres"`.

```bash
# SQLite (default) -- no setup required
python server.py

# Postgres
export DB_BACKEND=postgres
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=rag_dev
export POSTGRES_USER=rag_dev
export POSTGRES_PASSWORD=your_password
python server.py
```

Both backends are supported behind one interface (`rag/storage/connection.py`); `rag/storage/db.py`'s queries are identical regardless of which is active. This works because the schema uses only `TEXT`/`INTEGER` columns (portable across both engines) and the two real differences — placeholder syntax (`?` vs `%s`) and connection setup — are handled entirely inside `connection.py`.

**Limits of this approach, stated plainly**: this is a lightweight adapter, not an ORM. It does not handle schema migrations, connection pooling, or SQL dialect differences beyond placeholders and multi-statement scripts. If the schema grows more complex, reaching for a real query layer (SQLAlchemy or similar) would be the more robust path. Also note that the SQLite/Postgres branch in `connection.py` is decided once, at import time — switching `DB_BACKEND` at runtime after the process has started has no effect.

**On credentials**: the `POSTGRES_PASSWORD` default in `config.py` is a placeholder for local development only. It is read from an environment variable with a fallback value — adequate to avoid a hardcoded secret in source, but not a substitute for real secret management (a secrets manager, or an environment with no fallback default at all) in any shared or production environment.

### Vector search backend: in-process FAISS or Qdrant

Set via `config.VECTOR_BACKEND`: `"faiss"` (default, in-process) or `"qdrant"` (separate service, matching the same architectural pattern as Postgres/Redis).

```bash
docker compose up -d
export VECTOR_BACKEND=qdrant
python server.py
```

Two real, verified advantages over the FAISS wrapper: Qdrant accepts our UUID `chunk_id`s directly as point IDs (no `int_id` mapping layer needed at all, unlike `rag/storage/indexing.py`'s FAISS wrapper), and it supports **true deletion** rather than tombstoning — Qdrant handles reclaiming that space internally, not something this codebase needs to reason about for that backend. `save()`/`load_from_disk()` are no-ops for Qdrant (it persists continuously on its own, as long as its data directory is bind-mounted — see `docker-compose.yml`), which is why `server.py`'s `lifespan` needs no Qdrant-specific branching at all.

**One constraint, verified directly**: Qdrant point IDs must be either unsigned integers or genuine UUID strings — this is naturally satisfied since `rag/ingestion/pipeline.py` always generates real `str(uuid.uuid4())` chunk_ids, but worth knowing if you ever extend the ID scheme.

**A real risk worth knowing, not hidden**: `vector_index` is a singleton, constructed exactly once when `rag/storage/indexing.py` is first imported — unlike `rag/storage/connection.py`'s `Connection`, which is created fresh on every call. If `VECTOR_BACKEND=qdrant` is set but Qdrant isn't actually reachable at that first-import moment, the connection attempt fails immediately and the whole application (or test collection) crashes at startup, rather than degrading gracefully. This mirrors how Postgres/Redis behave if misconfigured too — not unique to Qdrant — but worth knowing before setting this in an environment where Qdrant's availability isn't guaranteed.

### Vector index type: HNSW or flat

Set via `config.INDEX_TYPE`: `"hnsw"` (default, approximate) or `"flat"` (exact brute-force). HNSW tuning knobs (only used when `INDEX_TYPE == "hnsw"`): `HNSW_M` (graph connectivity), `HNSW_EF_CONSTRUCTION` (build-time search depth), `HNSW_EF_SEARCH` (query-time search depth).

**A real, permanent trade-off worth understanding, not a bug**: FAISS's HNSW does not support true removal (verified directly — calling `remove_ids()` on it raises `RuntimeError`). `VectorIndex.remove()` falls back to deleting the entry from our own chunk_id mapping only — the vector becomes permanently unreachable through `search()`, but physically remains in the graph forever. This is the same pattern real vector databases use (Qdrant tombstones within sealed HNSW segments; Milvus marks vectors invalid until compaction) — not something unique or wrong about this implementation. The cost: tombstoned vectors accumulate over time, wasting memory and slightly slowing graph traversal, with no way to reclaim that space short of a full rebuild (see Planned enhancements).

**A real bug that WAS found and fixed here (worth knowing the shape of, in case it resurfaces elsewhere)**: `faiss.IndexHNSWFlat`'s constructor defaults to L2 distance if no metric is specified — unlike `IndexFlatIP`, which is inner product by construction. This silently broke `rag/dedup/semantic.py`'s duplicate check (`score >= SEMANTIC_DUP_COSINE_THRESHOLD`): with L2 distance, a *large* value means genuinely dissimilar vectors, but the threshold check assumed larger meant *more* similar — so unrelated documents could get misread as semantic duplicates of whatever was ingested first, silently excluding them from the vector index entirely. Confirmed directly (HNSW returned ~0.18-scale scores against Flat's ~0.90-scale for the identical query/vectors) before fixing by passing `faiss.METRIC_INNER_PRODUCT` explicitly to the `IndexHNSWFlat` constructor.

### Vector index persistence

Without persistence, every server restart re-runs the embedding model over every indexable chunk in the entire corpus before it can serve a single request — fine at toy scale, a severe problem at millions of chunks (a restart could take hours). `rag/storage/indexing.py`'s `VectorIndex.save()`/`load_from_disk()` persist the FAISS index and its chunk_id mapping to `config.FAISS_INDEX_PATH` (`data/faiss_index.faiss` + `data/faiss_index.meta.json` by default). `server.py`'s `lifespan` tries loading this on startup, only falling back to a full rebuild-from-database if nothing was persisted yet (first run, or the files were deleted) — and the pipeline saves again after every ingestion that actually changes the index, so a later restart never re-embeds anything from before.

The near-duplicate index gets the equivalent treatment: the `"memory"` backend is rebuilt at startup too (reconstructed from stored parent chunks, since raw document text isn't persisted separately — see `get_all_document_texts_for_near_dedup()`'s docstring for the caveat on exactness), and the `"redis"` backend persists on its own as long as Redis's data directory is bind-mounted (see `docker-compose.yml`).

### PDF/image extraction method: PyMuPDF or Docling

Set via `config.PDF_EXTRACTION_METHOD` (or the `PDF_EXTRACTION_METHOD` environment variable): `"pymupdf"` (default) or `"docling"`.

```bash
export PDF_EXTRACTION_METHOD=docling
python server.py
```

Docling ([MIT-licensed](https://github.com/DS4SD/docling), IBM-developed, now under the Linux Foundation's Agentic AI Foundation) outputs Markdown with tables preserved as real Markdown tables — automatically picked up by the existing table-detection logic in `rag/ingestion/tables.py`, no extra code needed. It's also the only option here that supports image files (`.png`/`.jpg`/`.jpeg`, via OCR) — PyMuPDF handles PDFs only. The tradeoff: meaningfully heavier and slower per document, since it runs real ML models (layout detection, table structure recognition, OCR) rather than pure text extraction.

**Not verified end-to-end by the author** — this was implemented against Docling's documented API but the dependency was too large to install in the development environment. `tests/test_docling_integration.py` runs a real extraction if `docling` is installed (skips otherwise); `manual_test_docling.py` is a standalone script for testing against your own PDFs directly (it's a pure function — no database or index is touched, nothing to clean up). Verify both yourself before relying on this in real use.

### Near-duplicate detection backend: in-memory or Redis

Set via `config.NEAR_DUP_BACKEND` (or the `NEAR_DUP_BACKEND` environment variable): `"memory"` (default) or `"redis"`.

```bash
# In-memory (default) -- no setup required
python server.py

# Redis
export NEAR_DUP_BACKEND=redis
export REDIS_HOST=localhost
export REDIS_PORT=6379
python server.py
```

The Redis backend uses datasketch's own built-in Redis storage layer (`MinHashLSH(storage_config={"type": "redis", ...})`) rather than a hand-rolled implementation -- it's the same `MinHashLSH` class as the "memory" backend, just pointed at Redis, so it reuses datasketch's own optimal band/row parameter search internally rather than an approximation. One detail that matters and is easy to miss: `NEAR_DUP_REDIS_BASENAME` must stay fixed -- datasketch generates a random one if omitted, which would silently give every process its own disconnected index (verified this failure mode directly before settling on a fixed value).

### Running a local Postgres instance for testing

```bash
docker compose up -d
```

This starts both Postgres and Redis, using `docker-compose.yml` (included in the repo). Postgres's data is bind-mounted to `data/postgres_data/` on disk -- unlike a plain `docker run` without a volume, this means the data survives container removal (`docker compose down`, even `docker compose down -v`, since a bind mount isn't a Docker-managed volume). The only way to actually delete it is `rm -rf data/postgres_data`. Redis's data is not persisted across container removal by design (see the comment in `docker-compose.yml`) -- acceptable since the near-duplicate index is meant to be rebuildable as documents are re-ingested.

The container has no tables until the app runs and creates them:

```bash
export DB_BACKEND=postgres
python server.py
```

Watch the startup output — `init_db()` runs `CREATE TABLE IF NOT EXISTS` against Postgres on first launch. Only after this will a database client show any tables.

### Visualizing database contents

- **SQLite**: VSCode "SQLite Viewer" extension, [DB Browser for SQLite](https://sqlitebrowser.org/), or `sqlite3 data/rag.db "..."`.
- **Postgres**: VSCode "SQLTools" (`mtxr.sqltools`) with the PostgreSQL driver (`mtxr.sqltools-driver-pg`). Connect with: host `localhost`, port `5432`, database `rag_dev`, username `rag_dev`, password `rag_dev_password` (matching `docker-compose.yml`'s defaults).
- **Either**: [DBeaver](https://dbeaver.io/) (free) supports both SQLite and Postgres in one tool.

## Concept reference

| Concept | Module | Key function |
|---|---|---|
| Fixed-size chunking with overlap | `rag/ingestion/chunking.py` | `_split_fixed()` |
| Semantic chunking | `rag/ingestion/chunking.py` | `_split_semantic()` |
| Parent/child chunk structure | `rag/ingestion/chunking.py`, `rag/storage/db.py` | `chunk_document()`, `get_chunks_with_parent_by_ids()` |
| Multi-format extraction | `rag/ingestion/extractors.py` | `EXTRACTORS` registry |
| Configurable PDF/image extraction (PyMuPDF or Docling) | `rag/ingestion/extractors.py` | `_extract_pdf()`, `_extract_with_docling()` |
| Table detection and embedding description | `rag/ingestion/tables.py` | `split_table_blocks()`, `get_embedding_text()` |
| Configurable storage backend (SQLite/Postgres) | `rag/storage/connection.py` | `Connection`, `get_connection()` |
| Exact duplicate detection (filename-independent) | `rag/storage/db.py` | `get_document_by_content_hash()` |
| Near-duplicate detection (MinHash/LSH, configurable backend) | `rag/dedup/near.py` | `check_near_duplicate()`, `_MemoryBackend`, `_RedisBackend` (datasketch's native Redis storage) |
| Semantic duplicate detection | `rag/dedup/semantic.py` | `find_semantic_duplicate()` |
| Pipeline orchestration | `rag/ingestion/pipeline.py` | `ingest_document()` |
| Versioning and soft-delete | `rag/storage/db.py`, `rag/ingestion/pipeline.py` | `mark_document_stale()` |
| Vector removal | `rag/storage/indexing.py` | `VectorIndex.remove()` |
| Vector search (HNSW default, or exact flat) | `rag/storage/indexing.py` | `build_base_index()` |
| HNSW tombstone-based soft-delete | `rag/storage/indexing.py` | `VectorIndex.remove()`, `search()` (over-fetch + filter) |
| Configurable vector backend (FAISS in-process or Qdrant service) | `rag/storage/indexing.py`, `rag/storage/qdrant_index.py` | `_build_vector_index()`, `QdrantVectorIndex` |
| Vector index persistence (avoids re-embedding on restart) | `rag/storage/indexing.py`, `server.py` | `VectorIndex.save()`, `load_from_disk()`, `lifespan()` |

## Continuous integration

`.github/workflows/tests.yml` runs the full test suite on every push and pull request, with real Postgres and Redis service containers (matching `config.py`'s default connection settings exactly, so no extra environment configuration is needed) — this is what makes `test_postgres_integration.py` and `test_redis_integration.py` actually run in CI, not just skip. `docling` is included in `requirements.txt`, so `test_docling_integration.py` runs for real too.

## Testing

```bash
pytest tests/ -v
```

The test suite mirrors the module structure:

| Test file | Covers |
|---|---|
| `test_dedup_exact.py`, `test_dedup_near.py` | Each deduplication strategy in isolation |
| `test_chunking.py`, `test_tables.py`, `test_extractors.py` | Ingestion-phase logic (pure functions, no I/O) |
| `test_storage_db.py`, `test_storage_indexing.py` | Document store and vector index, including versioning and vector removal |
| `test_postgres_integration.py` | The one file that runs against a REAL Postgres instance rather than the forced-SQLite default (see below) |
| `test_redis_integration.py` | The one file that runs against a REAL Redis instance rather than the forced-in-memory default (see below) |
| `test_qdrant_integration.py` | The one file that runs against a REAL Qdrant instance rather than the forced-FAISS default (see below) |
| `test_docling_integration.py` | Runs real Docling extraction if installed (skips otherwise) -- pure function, no cleanup needed |
| `test_ingestion_pipeline.py` | End-to-end pipeline behavior (dedup, versioning) |
| `test_retrieval.py` | Retrieval mechanics |
| `test_api.py` | HTTP-level tests against the running FastAPI application |

Most tests use a `fake_embeddings` fixture (see `tests/conftest.py`) that returns deterministic vectors instead of loading a real model, keeping the suite fast. These fixtures are not semantically meaningful and are not suited to evaluating retrieval quality — only real-model tests can do that. Every test **except** `test_postgres_integration.py` receives a fresh database, vector index, and MinHash/LSH index via autouse fixtures, with the storage backend forced to SQLite regardless of the real environment's `DB_BACKEND` setting — this keeps the default test run fast and independent of any running database service.

**Testing the Postgres backend specifically**: `test_postgres_integration.py` is the one file that deliberately overrides that SQLite-forcing and connects to a real Postgres instance. It skips itself (not a failure) if Postgres isn't reachable, so it's safe to always include in a normal test run:

```bash
docker compose up -d               # start Postgres and Redis first
pytest tests/test_postgres_integration.py -v
```

**Testing the Redis backend specifically**: `test_redis_integration.py` follows the same pattern — deliberately overrides the forced in-memory MinHash backend, connects to a real Redis instance (using logical DB 15, flushed before/after each test, isolated from any real data in the default DB), and skips cleanly if Redis isn't reachable.

```bash
docker compose up -d
pytest tests/test_redis_integration.py -v
```

**Testing the Qdrant backend specifically**: `test_qdrant_integration.py` follows the same pattern — uses a dedicated test collection (`rag_chunks_test`, deleted before and after each test), and skips cleanly if Qdrant isn't reachable.

```bash
docker compose up -d
pytest tests/test_qdrant_integration.py -v
```

**Testing Docling extraction specifically**: `test_docling_integration.py` skips if `docling` isn't installed, but it's included in `requirements.txt` by default, so it should just run:

```bash
pytest tests/test_docling_integration.py -v
```

For manually testing against your own real PDFs (including ones with actual tables, which the test's synthetic sample doesn't have):

```bash
python manual_test_docling.py /path/to/your/document.pdf
```

## Planned enhancements

Deliberately deferred work, kept here so intent isn't lost between sessions:

| Item | Why it matters | Notes |
|---|---|---|
| Periodic full HNSW rebuild | Tombstoned vectors accumulate in the graph forever (HNSW can't truly remove them), wasting memory and slightly slowing traversal over time — the same reason Qdrant/Milvus run background compaction | Rebuild by re-adding only the live vectors into a fresh index; could run on a schedule or be triggered once tombstone count crosses a threshold |
| Batch vector index writes during bulk ingestion | `/seed` calls `ingest_document()` individually per passage, saving the index to disk once per document (up to 500 separate disk writes for a 500-passage seed run) — real, avoidable overhead at bulk-load scale | Defer the save until the whole bulk operation finishes, or build the index once at the end rather than incrementally during load (the same "load first, index after" pattern databases use for bulk imports) |
| Hybrid retrieval (dense + BM25) | Pure embedding search misses exact keyword/ID/code matches | Combine with reciprocal rank fusion or similar |
| Reranking | Cheap ANN retrieval casts a wide net; a cross-encoder reranker picks the best few from it | Two-stage "cheap then precise" pattern |
| Query rewriting | Raw user queries are often short/ambiguous; rewriting before retrieval improves match quality | — |
| Run Docling as a separate service, not in-process | `PDF_EXTRACTION_METHOD=docling` loads its ML models directly into the API server process — memory footprint on every replica, workers blocked for the duration of each conversion (seconds to tens of seconds) | `docling-serve` (Docling's own officially maintained microservice, ready-made container images, async job queue) is the natural fix — not a small change (sync in-process call → async HTTP call with timeout/job-polling handling), but well-supported by existing official tooling |
| PySpark-based batch near-duplicate detection | `rag/dedup/near.py`'s "memory"/"redis" backends suit real-time, per-upload checking; a different tier of solution is needed for periodic full-corpus re-scans across millions of documents | Likely coexists with, rather than replaces, the real-time backends |
| File storage | Raw uploaded files are processed and discarded, never persisted — no source-document display, no re-processing without re-upload, no compliance retention, no debugging bad extractions | Planned as a `FileStorage` interface mirroring `DB_BACKEND`: `LocalDiskBackend` (`data/uploads/`) and `S3Backend`, selected via config |
| A real UI | Currently Swagger/curl only | Depends on file storage being in place first (for source-document display) |
| Chat session management + context compaction | No conversation/session concept exists — each `/chat` call is stateless | Needs session storage, multi-turn context, and a truncation/summarization strategy once history grows too long |
| Multi-vendor generation support | `rag/generation.py` calls Anthropic's API directly | Abstract behind a common interface, swappable via config — same spirit as the storage backend abstraction |
| Secrets management | `config.py`'s Postgres password falls back to a hardcoded dev default if no env var is set — fine locally, not for any shared/production environment | Remove the fallback default and/or integrate a real secrets manager |

## Known limitations

- Vector search defaults to `"hnsw"` (approximate, `config.INDEX_TYPE`); `"flat"` (exact brute-force) is still available. HNSW cannot truly remove a vector (verified directly against FAISS — `remove_ids()` raises `RuntimeError` for this index type), so soft-deleted vectors are tombstoned (permanently unreachable through search, but still physically occupying space in the graph) rather than genuinely erased — the same pattern real vector databases (Qdrant, Milvus) use, requiring periodic background compaction/rebuild to reclaim that space. Not yet implemented here — see Planned enhancements. The index is persisted to disk (`config.FAISS_INDEX_PATH`) and loaded on startup rather than re-embedded from scratch every restart — see Configuration below.
- Retrieval is dense-only; no hybrid (BM25) retrieval, reranking, or query rewriting yet.
- Table detection works on already-Markdown-formatted tables. PDF/image extraction now supports Docling (`config.PDF_EXTRACTION_METHOD = "docling"`), which outputs tables as real Markdown — picked up automatically by the existing table detection — but this path was implemented against Docling's documented API and has **not been run end-to-end** (it's too large a dependency to install in the environment this was developed in). Verify it directly before relying on it. The default `"pymupdf"` method remains raw-text-only, with no table structure and no image support at all.
- Document versioning uses filename as the identity key; a production system would use a stable external identifier.
- The near-duplicate index is rebuilt at startup for the `"memory"` backend (reconstructed from stored parent chunks, since raw document text isn't stored separately) and persists on its own for the `"redis"` backend (as long as Redis's data directory is bind-mounted — see `docker-compose.yml`). A Spark-based batch alternative for full-corpus scans is still planned (see Planned enhancements).
- The SQLite/Postgres storage adapter (`rag/storage/connection.py`) does not handle schema migrations, connection pooling, or SQL dialect differences beyond placeholders — see the Configuration section above.

## Troubleshooting

**`Fatal Python error: Segmentation fault` inside `faiss/swigfaiss.py`**

This is a known compatibility issue between `faiss-cpu` and `torch` (pulled in by `sentence-transformers`) on some platforms: both bundle their own OpenMP runtime, and loading both in one process can crash when FAISS runs its parallel search. The fix is already applied in `server.py` and `tests/conftest.py` (environment variables set before any other import) and in `rag/storage/indexing.py` (FAISS restricted to a single thread). If it still occurs:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
```

The more robust long-term fix is installing `faiss-cpu` from `conda-forge` rather than the PyPI wheel, since conda coordinates shared native libraries across packages instead of each bundling its own copy.