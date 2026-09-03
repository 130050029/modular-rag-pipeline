"""
Central configuration for the modular RAG pipeline.

Configuration follows this precedence:

    environment variable -> sensible local-development default

Paths are anchored to the repository so behavior does not depend on the
directory from which the application is launched.
"""

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Runtime safety
# ---------------------------------------------------------------------------
# Must happen before FAISS / torch / sentence-transformers are imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)

    if value is None:
        return default

    value = value.strip().lower()

    if value in {"1", "true", "yes", "on"}:
        return True

    if value in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"Invalid boolean value for {name}: {value!r}. "
        "Expected true/false."
    )


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid integer value for {name}: {value!r}"
        ) from exc


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)

    if value is None:
        return default

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid float value for {name}: {value!r}"
        ) from exc


def _require_positive(name: str, value: int | float) -> int | float:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _require_non_negative(name: str, value: int | float) -> int | float:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# config.py lives in:
#
#     <repo>/rag_python/config.py
#
# Therefore:
#
#     PROJECT_ROOT = <repo>
#     DATA_DIR     = <repo>/data
#
# This avoids relying on the caller's current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DB_PATH = os.environ.get(
    "DB_PATH",
    str(DATA_DIR / "rag.db"),
)

FAISS_INDEX_PATH = os.environ.get(
    "FAISS_INDEX_PATH",
    str(DATA_DIR / "faiss_index"),
)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
DB_BACKEND = os.environ.get("DB_BACKEND", "sqlite").strip().lower()

VALID_DB_BACKENDS = {"sqlite", "postgres"}

if DB_BACKEND not in VALID_DB_BACKENDS:
    raise ValueError(
        f"Unsupported DB_BACKEND={DB_BACKEND!r}. "
        f"Expected one of: {sorted(VALID_DB_BACKENDS)}"
    )


# PostgreSQL settings.
#
# These defaults are intentionally local-development values. Real deployment
# environments should provide them through environment variables / secret
# management rather than relying on defaults.
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = _require_positive(
    "POSTGRES_PORT",
    _env_int("POSTGRES_PORT", 5432),
)
POSTGRES_DB = os.environ.get("POSTGRES_DB", "rag_dev")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "rag_dev")
POSTGRES_PASSWORD = os.environ.get(
    "POSTGRES_PASSWORD",
    "rag_dev_password",
)


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME",
    "all-MiniLM-L6-v2",
)

EMBEDDING_DIM = _require_positive(
    "EMBEDDING_DIM",
    _env_int("EMBEDDING_DIM", 384),
)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
PARENT_CHUNK_SIZE_WORDS = _require_positive(
    "PARENT_CHUNK_SIZE_WORDS",
    _env_int("PARENT_CHUNK_SIZE_WORDS", 300),
)

SMALL_CHUNK_SIZE_WORDS = _require_positive(
    "SMALL_CHUNK_SIZE_WORDS",
    _env_int("SMALL_CHUNK_SIZE_WORDS", 100),
)

SMALL_CHUNK_OVERLAP_WORDS = _require_non_negative(
    "SMALL_CHUNK_OVERLAP_WORDS",
    _env_int("SMALL_CHUNK_OVERLAP_WORDS", 20),
)

CHUNKING_STRATEGY = os.environ.get(
    "CHUNKING_STRATEGY",
    "fixed",
).strip().lower()

VALID_CHUNKING_STRATEGIES = {"fixed", "semantic"}

if CHUNKING_STRATEGY not in VALID_CHUNKING_STRATEGIES:
    raise ValueError(
        f"Unsupported CHUNKING_STRATEGY={CHUNKING_STRATEGY!r}. "
        f"Expected one of: {sorted(VALID_CHUNKING_STRATEGIES)}"
    )

SEMANTIC_CHUNK_SIMILARITY_DROP = _env_float(
    "SEMANTIC_CHUNK_SIMILARITY_DROP",
    0.35,
)

SEMANTIC_CHUNK_MAX_WORDS = _require_positive(
    "SEMANTIC_CHUNK_MAX_WORDS",
    _env_int("SEMANTIC_CHUNK_MAX_WORDS", 200),
)


# ---------------------------------------------------------------------------
# PDF / image extraction
# ---------------------------------------------------------------------------
PDF_EXTRACTION_METHOD = os.environ.get(
    "PDF_EXTRACTION_METHOD",
    "pymupdf",
).strip().lower()

VALID_PDF_EXTRACTION_METHODS = {"pymupdf", "docling"}

if PDF_EXTRACTION_METHOD not in VALID_PDF_EXTRACTION_METHODS:
    raise ValueError(
        f"Unsupported PDF_EXTRACTION_METHOD={PDF_EXTRACTION_METHOD!r}. "
        f"Expected one of: {sorted(VALID_PDF_EXTRACTION_METHODS)}"
    )


# ---------------------------------------------------------------------------
# Table handling
# ---------------------------------------------------------------------------
TABLE_EMBEDDING_DESCRIPTION = os.environ.get(
    "TABLE_EMBEDDING_DESCRIPTION",
    "template",
).strip().lower()

VALID_TABLE_EMBEDDING_DESCRIPTIONS = {"template", "llm"}

if TABLE_EMBEDDING_DESCRIPTION not in VALID_TABLE_EMBEDDING_DESCRIPTIONS:
    raise ValueError(
        f"Unsupported TABLE_EMBEDDING_DESCRIPTION="
        f"{TABLE_EMBEDDING_DESCRIPTION!r}. "
        f"Expected one of: {sorted(VALID_TABLE_EMBEDDING_DESCRIPTIONS)}"
    )


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
MINHASH_NUM_PERM = _require_positive(
    "MINHASH_NUM_PERM",
    _env_int("MINHASH_NUM_PERM", 128),
)

MINHASH_SHINGLE_SIZE = _require_positive(
    "MINHASH_SHINGLE_SIZE",
    _env_int("MINHASH_SHINGLE_SIZE", 9),
)

NEAR_DUP_JACCARD_THRESHOLD = _env_float(
    "NEAR_DUP_JACCARD_THRESHOLD",
    0.8,
)

SEMANTIC_DUP_COSINE_THRESHOLD = _env_float(
    "SEMANTIC_DUP_COSINE_THRESHOLD",
    0.95,
)


# ---------------------------------------------------------------------------
# Near-duplicate backend
# ---------------------------------------------------------------------------
NEAR_DUP_BACKEND = os.environ.get(
    "NEAR_DUP_BACKEND",
    "memory",
).strip().lower()

VALID_NEAR_DUP_BACKENDS = {"memory", "redis"}

if NEAR_DUP_BACKEND not in VALID_NEAR_DUP_BACKENDS:
    raise ValueError(
        f"Unsupported NEAR_DUP_BACKEND={NEAR_DUP_BACKEND!r}. "
        f"Expected one of: {sorted(VALID_NEAR_DUP_BACKENDS)}"
    )

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")

REDIS_PORT = _require_positive(
    "REDIS_PORT",
    _env_int("REDIS_PORT", 6379),
)

REDIS_DB = _require_non_negative(
    "REDIS_DB",
    _env_int("REDIS_DB", 0),
)

NEAR_DUP_REDIS_BASENAME = os.environ.get(
    "NEAR_DUP_REDIS_BASENAME",
    "modular_rag_pipeline_near_dup_lsh",
).encode("utf-8")


# ---------------------------------------------------------------------------
# Vector index
# ---------------------------------------------------------------------------
INDEX_TYPE = os.environ.get(
    "INDEX_TYPE",
    "hnsw",
).strip().lower()

VALID_INDEX_TYPES = {"flat", "hnsw"}

if INDEX_TYPE not in VALID_INDEX_TYPES:
    raise ValueError(
        f"Unsupported INDEX_TYPE={INDEX_TYPE!r}. "
        f"Expected one of: {sorted(VALID_INDEX_TYPES)}"
    )


# ---------------------------------------------------------------------------
# Vector backend
# ---------------------------------------------------------------------------
VECTOR_BACKEND = os.environ.get(
    "VECTOR_BACKEND",
    "faiss",
).strip().lower()

VALID_VECTOR_BACKENDS = {"faiss", "qdrant"}

if VECTOR_BACKEND not in VALID_VECTOR_BACKENDS:
    raise ValueError(
        f"Unsupported VECTOR_BACKEND={VECTOR_BACKEND!r}. "
        f"Expected one of: {sorted(VALID_VECTOR_BACKENDS)}"
    )

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")

QDRANT_PORT = _require_positive(
    "QDRANT_PORT",
    _env_int("QDRANT_PORT", 6333),
)

QDRANT_COLLECTION = os.environ.get(
    "QDRANT_COLLECTION",
    "rag_chunks",
)


# ---------------------------------------------------------------------------
# HNSW tuning
# ---------------------------------------------------------------------------
HNSW_M = _require_positive(
    "HNSW_M",
    _env_int("HNSW_M", 32),
)

HNSW_EF_CONSTRUCTION = _require_positive(
    "HNSW_EF_CONSTRUCTION",
    _env_int("HNSW_EF_CONSTRUCTION", 200),
)

HNSW_EF_SEARCH = _require_positive(
    "HNSW_EF_SEARCH",
    _env_int("HNSW_EF_SEARCH", 64),
)


# ---------------------------------------------------------------------------
# Query intelligence
# ---------------------------------------------------------------------------
# When enabled, the user query is rewritten once before retrieval.
# Disabled by default so existing retrieval behavior remains unchanged.
QUERY_REWRITE_ENABLED = (
    os.environ.get("QUERY_REWRITE_ENABLED", "false").lower() == "true"
)
QUERY_EXPANSION_ENABLED = (
    os.environ.get("QUERY_EXPANSION_ENABLED", "false").lower() == "true"
)

QUERY_EXPANSION_MAX_QUERIES = int(
    os.environ.get("QUERY_EXPANSION_MAX_QUERIES", "3")
)

QUERY_DECOMPOSITION_ENABLED = (
    os.environ.get("QUERY_DECOMPOSITION_ENABLED", "false").lower() == "true"
)

QUERY_DECOMPOSITION_MAX_QUERIES = 4
# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = _require_positive(
    "TOP_K",
    _env_int("TOP_K", 4),
)

SEARCH_MODE = os.environ.get(
    "SEARCH_MODE",
    "hybrid",
).strip().lower()

VALID_SEARCH_MODES = {"dense", "sparse", "hybrid"}

if SEARCH_MODE not in VALID_SEARCH_MODES:
    raise ValueError(
        f"Unsupported SEARCH_MODE={SEARCH_MODE!r}. "
        f"Expected one of: {sorted(VALID_SEARCH_MODES)}"
    )

RRF_K = _require_positive(
    "RRF_K",
    _env_int("RRF_K", 60),
)

DENSE_CANDIDATE_K = _require_positive(
    "DENSE_CANDIDATE_K",
    _env_int("DENSE_CANDIDATE_K", 20),
)

SPARSE_CANDIDATE_K = _require_positive(
    "SPARSE_CANDIDATE_K",
    _env_int("SPARSE_CANDIDATE_K", 20),
)


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------
RERANK_ENABLED = _env_bool(
    "RERANK_ENABLED",
    False,
)

RERANKER_MODEL = os.environ.get(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

RERANK_CANDIDATE_K = _require_positive(
    "RERANK_CANDIDATE_K",
    _env_int("RERANK_CANDIDATE_K", 20),
)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.environ.get(
    "LLM_PROVIDER",
    "ollama",
).strip().lower()

VALID_LLM_PROVIDERS = {"ollama", "anthropic"}

if LLM_PROVIDER not in VALID_LLM_PROVIDERS:
    raise ValueError(
        f"Unsupported LLM_PROVIDER={LLM_PROVIDER!r}. "
        f"Expected one of: {sorted(VALID_LLM_PROVIDERS)}"
    )

LLM_MAX_TOKENS = _require_positive(
    "LLM_MAX_TOKENS",
    _env_int("LLM_MAX_TOKENS", 500),
)

LLM_TIMEOUT = _require_positive(
    "LLM_TIMEOUT",
    _env_int("LLM_TIMEOUT", 120),
)


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
).rstrip("/")

OLLAMA_MODEL = os.environ.get(
    "OLLAMA_MODEL",
    "qwen2.5:0.5b",
)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------
ANTHROPIC_MODEL = os.environ.get(
    "ANTHROPIC_MODEL",
    "claude-sonnet-4-5-20250929",
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------
CONTEXT_MAX_CHARS = _require_positive(
    "CONTEXT_MAX_CHARS",
    _env_int("CONTEXT_MAX_CHARS", 12000),
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
HOST = os.environ.get("HOST", "127.0.0.1")

PORT = _require_positive(
    "PORT",
    _env_int("PORT", 8000),
)