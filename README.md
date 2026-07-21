# Modular RAG Pipeline

A modular, end-to-end Retrieval-Augmented Generation (RAG) pipeline built with FastAPI, FAISS, and sentence-transformers. It implements document ingestion with multi-format extraction, a three-layer deduplication strategy, document versioning with soft-delete, parent/child chunking, and table-aware embedding — each concern isolated into its own module so individual components can be understood, tested, and replaced independently.

## Features

- **Multi-format ingestion**: plain text, Markdown, HTML, and PDF, dispatched through a pluggable extractor registry
- **Three-layer deduplication**: exact (content hash), near-duplicate (MinHash + LSH), and semantic (embedding similarity)
- **Document versioning**: re-ingesting an updated document soft-deletes the previous version and removes its vectors from the live index
- **Parent/child chunking**: small chunks are embedded and searched; their parent chunks are what get passed to the LLM, keeping retrieval precise and generated context complete
- **Two configurable chunking strategies**: fixed-size word windows, or semantic boundary detection based on sentence-embedding similarity
- **Table-aware embedding**: tables are detected and embedded via a generated natural-language description rather than their raw structure
- **FAISS-backed vector search** with support for removal (required for soft-delete)
- **Full test suite** (pytest) covering every module in isolation plus end-to-end integration tests

## Architecture

```
config.py              Every tunable value in the project; nothing else hardcodes settings
server.py               FastAPI application (thin wiring layer only)
requirements.txt
rag/
├── storage/            Document store (SQLite) and vector index (FAISS)
│   ├── db.py
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

All configurable values live in `config.py`, including the database path, embedding model, chunking strategy and sizes, and deduplication thresholds. No other file hardcodes these values.

**Note on storage backend**: `config.DB_PATH` controls which SQLite file is used, but switching to a different database engine entirely (e.g. Postgres) requires a code change in addition to configuration — `rag/storage/db.py` connects via `sqlite3` directly, and a different engine uses a different client library and connection model. That change is isolated to this single file; no other module touches storage directly.

## Concept reference

| Concept | Module | Key function |
|---|---|---|
| Fixed-size chunking with overlap | `rag/ingestion/chunking.py` | `_split_fixed()` |
| Semantic chunking | `rag/ingestion/chunking.py` | `_split_semantic()` |
| Parent/child chunk structure | `rag/ingestion/chunking.py`, `rag/storage/db.py` | `chunk_document()`, `get_chunks_with_parent_by_ids()` |
| Multi-format extraction | `rag/ingestion/extractors.py` | `EXTRACTORS` registry |
| Table detection and embedding description | `rag/ingestion/tables.py` | `split_table_blocks()`, `get_embedding_text()` |
| Exact duplicate detection (filename-independent) | `rag/storage/db.py` | `get_document_by_content_hash()` |
| Near-duplicate detection (MinHash/LSH) | `rag/dedup/near.py` | `check_near_duplicate()` |
| Semantic duplicate detection | `rag/dedup/semantic.py` | `find_semantic_duplicate()` |
| Pipeline orchestration | `rag/ingestion/pipeline.py` | `ingest_document()` |
| Versioning and soft-delete | `rag/storage/db.py`, `rag/ingestion/pipeline.py` | `mark_document_stale()` |
| Vector removal | `rag/storage/indexing.py` | `VectorIndex.remove()` |
| Vector search (exact, brute-force) | `rag/storage/indexing.py` | `build_base_index()` |

## Viewing the database

`rag.db` is a plain SQLite file. Options for inspecting it:
- VSCode: the "SQLite Viewer" extension
- [DB Browser for SQLite](https://sqlitebrowser.org/) (standalone GUI)
- CLI: `sqlite3 rag.db "SELECT * FROM chunks LIMIT 5;"`

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
| `test_ingestion_pipeline.py` | End-to-end pipeline behavior (dedup, versioning) |
| `test_retrieval.py` | Retrieval mechanics |
| `test_api.py` | HTTP-level tests against the running FastAPI application |

Most tests use a `fake_embeddings` fixture (see `tests/conftest.py`) that returns deterministic vectors instead of loading a real model, keeping the suite fast. These fixtures are not semantically meaningful and are not suited to evaluating retrieval quality — only real-model tests can do that. Every test receives a fresh database, vector index, and MinHash/LSH index via autouse fixtures, ensuring tests do not affect one another.

## Known limitations

- Vector search uses exact brute-force search (`IndexFlatIP`); an approximate index (HNSW/IVF) is a planned addition, isolated to `rag/storage/indexing.py`.
- Retrieval is dense-only; no hybrid (BM25) retrieval, reranking, or query rewriting yet.
- Table detection handles tables already present in Markdown form within a document; it does not reconstruct tables split across multiple PDF pages.
- Document versioning uses filename as the identity key; a production system would use a stable external identifier.
- The MinHash/LSH index in `rag/dedup/near.py` is in-memory only and rebuilt on every restart.

## Troubleshooting

**`Fatal Python error: Segmentation fault` inside `faiss/swigfaiss.py`**

This is a known compatibility issue between `faiss-cpu` and `torch` (pulled in by `sentence-transformers`) on some platforms: both bundle their own OpenMP runtime, and loading both in one process can crash when FAISS runs its parallel search. The fix is already applied in `server.py` and `tests/conftest.py` (environment variables set before any other import) and in `rag/storage/indexing.py` (FAISS restricted to a single thread). If it still occurs:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
```

The more robust long-term fix is installing `faiss-cpu` from `conda-forge` rather than the PyPI wheel, since conda coordinates shared native libraries across packages instead of each bundling its own copy.