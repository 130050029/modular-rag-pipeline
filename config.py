"""
config.py -- single source of truth for every tunable value in this project.

Every other file imports from here rather than hardcoding values, so that
when we experiment with a setting (chunk size, top_k, which index type),
there's exactly one place to change it.
"""

import os

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
# "sqlite" (default, zero-setup) or "postgres". See rag/storage/connection.py
# for how both are supported behind one interface -- db.py's queries never
# change regardless of which backend is active.
DB_BACKEND = os.environ.get("DB_BACKEND", "sqlite")

# Used only when DB_BACKEND == "sqlite".
DB_PATH = "data/rag.db"

# Persisted vector index location -- produces <FAISS_INDEX_PATH>.faiss and
# <FAISS_INDEX_PATH>.meta.json. Without this, every server restart re-runs
# the embedding model over every indexable chunk in the corpus, which at
# real scale (millions of chunks) turns a restart into a multi-hour
# operation. See rag/storage/indexing.py's save()/load_from_disk() and
# server.py's lifespan.
FAISS_INDEX_PATH = "data/faiss_index"

# Used only when DB_BACKEND == "postgres". These defaults are placeholders
# for local development ONLY -- never commit real credentials here. Proper
# secret management (reading exclusively from the environment, or a secrets
# manager, with no fallback default at all) is a planned follow-up; for now
# this at least keeps credentials out of hardcoded Python values and lets
# each environment override via its own env vars.
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.environ.get("POSTGRES_DB", "rag_dev")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "rag_dev")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "rag_dev_password")

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, CPU-friendly
EMBEDDING_DIM = 384

# ---------------------------------------------------------------------------
# Chunking -- now parent (large, shown to LLM) + small (precise, embedded)
# ---------------------------------------------------------------------------
PARENT_CHUNK_SIZE_WORDS = 300
SMALL_CHUNK_SIZE_WORDS = 100
SMALL_CHUNK_OVERLAP_WORDS = 20

# "fixed" = fixed-size word windows (current default)
# "semantic" = split where consecutive-sentence embedding similarity drops
CHUNKING_STRATEGY = "fixed"
SEMANTIC_CHUNK_SIMILARITY_DROP = 0.35   # similarity drop that triggers a new chunk boundary
SEMANTIC_CHUNK_MAX_WORDS = 200          # safety cap so one chunk can't grow unbounded

# ---------------------------------------------------------------------------
# PDF/image extraction method
# ---------------------------------------------------------------------------
# "pymupdf" (default) -- fast, lightweight, raw text only. Does not preserve
#   table structure, and cannot handle images at all.
# "docling" -- IBM's open-source (MIT) document intelligence toolkit. Slower
#   and pulls in real ML models (layout detection, table structure
#   recognition, OCR) as dependencies, but outputs Markdown WITH tables
#   preserved as real Markdown tables -- meaning they're automatically
#   picked up by our existing is_table_like()/split_into_segments() logic,
#   no extra code needed. Also the only option that supports images (via OCR).
PDF_EXTRACTION_METHOD = os.environ.get("PDF_EXTRACTION_METHOD", "pymupdf")

# ---------------------------------------------------------------------------
# Table handling
# ---------------------------------------------------------------------------
# "template" = cheap, deterministic description (column names + row count)
# "llm"      = ask Claude for a one-sentence description (higher quality, costs a call per table)
TABLE_EMBEDDING_DESCRIPTION = "template"


# ---------------------------------------------------------------------------
# Deduplication thresholds
# ---------------------------------------------------------------------------
MINHASH_NUM_PERM = 128          # number of hash functions in each MinHash signature
MINHASH_SHINGLE_SIZE = 9        # words per shingle (long enough to be a meaningful signal)
NEAR_DUP_JACCARD_THRESHOLD = 0.8    # doc-level near-duplicate cutoff (MinHash/LSH; "memory" backend only)
SEMANTIC_DUP_COSINE_THRESHOLD = 0.95  # chunk-level semantic-duplicate cutoff (embeddings)

# ---------------------------------------------------------------------------
# Near-duplicate detection backend
# ---------------------------------------------------------------------------
# "memory" (default) -- datasketch's MinHashLSH, in-process. Fast, zero
#   setup, but lost on restart and not shared across multiple app replicas.
# "redis" -- manual LSH banding stored in Redis sets. Persists across
#   restarts and is shared consistently across every app replica reading
#   the same Redis instance.
NEAR_DUP_BACKEND = os.environ.get("NEAR_DUP_BACKEND", "memory")

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))

# MUST be fixed and explicit -- datasketch generates a random basename if
# none is given, which means two separate processes would each silently get
# their own private, disconnected index, defeating the entire point of a
# shared Redis-backed index. Verified this failure mode directly against a
# real Redis instance before settling on this design.
NEAR_DUP_REDIS_BASENAME = b"modular_rag_pipeline_near_dup_lsh"


# ---------------------------------------------------------------------------
# Indexing / retrieval
# ---------------------------------------------------------------------------
# "flat" = exact brute-force (faiss.IndexFlatIP) -- fine at toy scale.
# Will add "hnsw" as a second option here when we cover ANN indexing.
INDEX_TYPE = "flat"
TOP_K = 4

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 8000