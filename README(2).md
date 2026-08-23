# Modular RAG Pipeline

A modular, end-to-end Retrieval-Augmented Generation (RAG) pipeline built with FastAPI, FAISS, and sentence-transformers. It implements document ingestion with multi-format extraction, a three-layer deduplication strategy, document versioning with soft-delete, parent/child chunking, and table-aware embedding — each concern isolated into its own module so individual components can be understood, tested, and replaced independently.

## Features

- **Multi-format ingestion**: plain text, Markdown, HTML, PDF, and (via Docling) images, dispatched through a pluggable extractor registry with a configurable PDF/image extraction method (PyMuPDF or Docling)
- **Three-layer deduplication**: exact (content hash), near-duplicate (MinHash + LSH), and semantic (embedding similarity)
- **Document versioning**: same source URI → new version; old version soft-deleted and its vectors removed from the search index
- **Parent/child chunking**: small chunks are embedded and retrieved; their larger parent chunks are returned to the LLM for more context
- **Table-aware chunking**: Markdown tables detected and described with an LLM-friendly embedding representation
- **Pluggable storage**: SQLite or PostgreSQL, selected via `DB_BACKEND`
- **Pluggable vector index**: FAISS (in-process) or Qdrant (service), selected via `VECTOR_BACKEND`
- **Configurable FAISS index**: HNSW (approximate, default) or exact flat search via `INDEX_TYPE`
- **Hybrid retrieval**: dense vector search + BM25/FTS keyword search fused with Reciprocal Rank Fusion (RRF)
- **Pluggable near-duplicate storage**: in-memory or Redis, selected via `NEAR_DUP_BACKEND`
- **Configurable PDF/image extraction**: PyMuPDF (default) or Docling via `PDF_EXTRACTION_METHOD`
- **Persistent vector index**: FAISS index saved to disk and loaded on restart, avoiding re-embedding the entire corpus
- **FastAPI API**: `/upload`, `/chat`, `/seed`, `/health`
- **Tests**: unit, integration, and API tests with isolated fixtures; real Postgres/Redis/Qdrant integration tests
- **CI**: GitHub Actions runs the test suite with real Postgres and Redis service containers

## Architecture

```text
                         ┌──────────────────────┐
                         │       FastAPI        │
                         │ /upload /chat /seed  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    RAG Pipeline      │
                         └──────────┬───────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                │                   │                   │
                ▼                   ▼                   ▼
        ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
        │  Extraction  │    │   Chunking   │    │    Dedup     │
        │ PDF/MD/HTML  │    │ Parent/Child │    │ Exact/Near/  │
        │ Text/Docling │    │ Tables       │    │ Semantic     │
        └──────────────┘    └──────────────┘    └──────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Embeddings      │
                         │ sentence-transformer │
                         └──────────┬───────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
                     ▼                             ▼
             ┌──────────────┐             ┌──────────────┐
             │ Vector Store │             │ Document DB  │
             │ FAISS/Qdrant │             │ SQLite/PG    │
             └──────┬───────┘             └──────────────┘
                    │
                    │
Query ──────────────┤
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
┌──────────────┐        ┌──────────────┐
│ Dense Search │        │ Sparse/BM25  │
│              │        │ FTS5 / PG    │
└──────┬───────┘        └──────┬───────┘
       │                       │
       └───────────┬───────────┘
                   ▼
             ┌───────────┐
             │    RRF    │
             └─────┬─────┘
                   │
                   ▼
            Parent Context
                   │
                   ▼
             ┌───────────┐
             │    LLM    │
             │  Claude   │
             └───────────┘
```

## Project structure

```text
modular-rag-pipeline/
├── config.py
├── server.py
├── eval_retrieval.py
├── requirements.txt
├── docker-compose.yml
├── README.md
├── bm25_brief.md
├── rag/
│   ├── generation.py
│   ├── embeddings.py
│   ├── retrieval.py
│   ├── rank_fusion.py
│   ├── ingestion/
│   │   ├── extractors.py
│   │   ├── chunking.py
│   │   ├── tables.py
│   │   └── pipeline.py
│   ├── dedup/
│   │   ├── exact.py
│   │   ├── near.py
│   │   └── semantic.py
│   └── storage/
│       ├── connection.py
│       ├── db.py
│       ├── indexing.py
│       └── qdrant_index.py
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_chunking.py
│   ├── test_dedup_exact.py
│   ├── test_dedup_near.py
│   ├── test_dedup_semantic.py
│   ├── test_extractors.py
│   ├── test_ingestion_pipeline.py
│   ├── test_keyword_search.py
│   ├── test_postgres_integration.py
│   ├── test_qdrant_integration.py
│   ├── test_redis_integration.py
│   ├── test_retrieval.py
│   ├── test_rank_fusion.py
│   ├── test_storage_db.py
│   ├── test_storage_indexing.py
│   └── test_tables.py
└── data/
    ├── manual_test_files/
    ├── golden_set/
    └── postgres_data/
```

## Quick start

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

The project uses environment variables for backend selection and credentials.

The defaults are deliberately chosen to make the learning project easy to run locally:

```text
DB_BACKEND=sqlite
VECTOR_BACKEND=faiss
INDEX_TYPE=hnsw
NEAR_DUP_BACKEND=memory
PDF_EXTRACTION_METHOD=pymupdf
HYBRID_SEARCH_ENABLED=true
```

See `config.py` for the complete list.

For Claude generation:

```bash
export ANTHROPIC_API_KEY="your-key"
```

### 4. Start the API

```bash
python server.py
```

The API will normally be available at:

```text
http://localhost:8000
```

Swagger/OpenAPI:

```text
http://localhost:8000/docs
```

### 5. Seed the sample corpus

```bash
curl -X POST http://localhost:8000/seed
```

### 6. Ask a question

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is discussed in the sample documents?"}'
```

## Backend configuration

The project intentionally keeps backend choices behind configuration so the
same application pipeline can be run with different implementations.

### Document database

```bash
export DB_BACKEND=sqlite
```

or:

```bash
export DB_BACKEND=postgres
```

### Vector backend

FAISS:

```bash
export VECTOR_BACKEND=faiss
```

Qdrant:

```bash
export VECTOR_BACKEND=qdrant
```

### FAISS index type

HNSW is the default:

```bash
export INDEX_TYPE=hnsw
```

Exact flat search:

```bash
export INDEX_TYPE=flat
```

### Near-duplicate backend

In-memory:

```bash
export NEAR_DUP_BACKEND=memory
```

Redis:

```bash
export NEAR_DUP_BACKEND=redis
export REDIS_HOST=localhost
export REDIS_PORT=6379
```

### PDF/image extraction

PyMuPDF:

```bash
export PDF_EXTRACTION_METHOD=pymupdf
```

Docling:

```bash
export PDF_EXTRACTION_METHOD=docling
```

Docling is particularly useful for documents where table structure matters.

## Hybrid retrieval

The retrieval layer currently combines two independent candidate generators:

```text
                         Query
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       Dense vector search        Sparse/BM25 search
          top-N candidates          top-N candidates
              │                         │
              └────────────┬────────────┘
                           ▼
                          RRF
                           │
                           ▼
                        top-K
                           │
                           ▼
                    Parent context
                           │
                           ▼
                          LLM
```

Dense and BM25 scores are deliberately **not** added together.

They are on incompatible scales. Instead, Reciprocal Rank Fusion uses only
the rank position of each result:

```text
RRF(d) = Σ 1 / (k + rank(d))
```

where `rank` is 1-based and `k` is the smoothing constant.

The default `RRF_K` is 60.

A result that appears near the top of both retrievers receives a strong
consensus score.

The implementation also protects against duplicate IDs appearing more than
once within a single ranked list. One retriever can contribute at most one
vote for a given chunk.

### BM25 implementation

SQLite uses FTS5 with its built-in `bm25()` function.

Postgres uses a generated `tsvector` column with `ts_rank()`.

These produce different raw score scales. This is expected and does not
matter to RRF because RRF consumes ranking positions rather than raw scores.

One SQLite-specific detail is important:

> FTS5's `bm25()` returns more-negative values for better matches.

Therefore SQLite results are ordered by BM25 score ascending.

User query text is tokenized and individual tokens are quoted before being
passed to FTS5. This prevents punctuation or arbitrary user input from
accidentally becoming FTS5 query syntax.

Only live, non-duplicate small chunks are included in keyword retrieval,
matching the definition of chunks that belong in the vector index.

## Running a local Postgres instance for testing

```bash
docker compose up -d
```

This starts both Postgres and Redis, using `docker-compose.yml`.

The container has no tables until the app runs and creates them:

```bash
export DB_BACKEND=postgres
python server.py
```

Watch the startup output — `init_db()` runs `CREATE TABLE IF NOT EXISTS` against
Postgres on first launch.

## Testing

```bash
pytest tests/ -v
```

Most tests use deterministic fake embeddings so that the suite remains fast
and independent of model downloads.

Backend-specific integration tests use real Postgres, Redis, and Qdrant where
available and skip cleanly when the service is unavailable.

## Continuous integration

`.github/workflows/tests.yml` runs the test suite on pushes and pull requests,
with real Postgres and Redis service containers.

## Production RAG learning roadmap

| Phase | Focus | Status |
|---|---|---|
| **A** | Retrieval quality: dense retrieval, BM25, RRF, golden-set design, Recall/MRR/NDCG, reranking, context selection | **Current** |
| **B** | Query intelligence: rewriting, expansion, multi-query, decomposition, routing | Planned |
| **C** | Generation quality: grounding, citations, structured output, refusal/insufficient-context handling | Planned |
| **D** | Evaluation: retrieval + answer-quality datasets, regression experiments, faithfulness/relevance/citation metrics | Planned |
| **E** | Production reliability: idempotency, retries, timeouts, background ingestion, index/database consistency, reconciliation | Planned |
| **F** | Observability: traces, latency, token/cost accounting, retrieval diagnostics, metrics | Planned |
| **G** | Security: prompt injection, indirect injection, authorization, tenant isolation, sensitive-data handling | Planned |
| **H** | Scale & deployment: workers, queues, connection pools, caching, load testing, Docker/cloud deployment, cost optimization | Planned |

## Known limitations

- HNSW uses tombstones for deletion; periodic compaction/rebuild is not yet implemented.
- Retrieval currently supports dense + BM25 hybrid search via RRF. Reranking and query rewriting are intentionally not implemented yet.
- The current evaluation set is a basic sanity set and is not yet sufficient for serious retrieval-quality measurement.
- Table-aware retrieval depends on the quality of upstream extraction.
- Document versioning and source identity are still simplified for learning purposes.
- Storage adapters do not yet provide production-grade migrations, pooling, or full SQL-dialect abstraction.
- Secrets management is still development-oriented.
- Raw uploaded files are not yet persisted in a dedicated file-storage abstraction.
- Chat sessions and multi-turn context management are not implemented.
