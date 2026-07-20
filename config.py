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
# This is the ONE line that changes if you point storage somewhere else --
# a different local path, a path outside the repo, or (with a corresponding
# change to rag/storage/db.py's connection logic -- see README) a real
# Postgres/other database on another machine entirely. Everything else in
# the codebase only ever imports DB_PATH from here, never hardcodes it.
DB_PATH = "data/rag.db"

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
NEAR_DUP_JACCARD_THRESHOLD = 0.8    # doc-level near-duplicate cutoff (MinHash/LSH)
SEMANTIC_DUP_COSINE_THRESHOLD = 0.95  # chunk-level semantic-duplicate cutoff (embeddings)


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