"""
conftest.py -- shared pytest fixtures.

Two important patterns worth understanding, not just copying:

1. "Patch where it's USED, not where it's DEFINED." rag/ingestion/pipeline.py
   does `from rag.embeddings import embed_texts` at import time -- that
   binds a name INSIDE pipeline.py's own namespace. Patching
   rag.embeddings.embed_texts after that import already happened does
   nothing to pipeline.py's copy of the name. So fixtures below patch each
   consuming module's own reference explicitly.

2. Fresh state per test. db.py, indexing.py's VectorIndex, and near.py's
   MinHash/LSH index all hold state (a file path, an in-memory index).
   Autouse fixtures reset all three before every test so tests can't leak
   into each other.
"""

# MUST run before faiss or torch/sentence-transformers get imported anywhere
# -- pytest imports every test file during collection, and whichever one
# happens to import rag.embeddings first will pull in torch; if that
# happens before this line runs, FAISS's search() can segfault. See README.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from pathlib import Path

# Ensure the project root (parent of tests/) is importable regardless of
# where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Point the database at a fresh temp SQLite file for every test."""
    db_file = tmp_path / "test_rag.db"
    monkeypatch.setattr("rag.storage.db.DB_PATH", str(db_file))
    from rag.storage.db import init_db
    init_db()
    yield str(db_file)


@pytest.fixture(autouse=True)
def fresh_vector_index(monkeypatch):
    """Swap in a brand-new, empty VectorIndex for every test -- and patch it
    into every module that imported the singleton directly."""
    from rag.storage.indexing import VectorIndex
    fresh = VectorIndex()
    monkeypatch.setattr("rag.storage.indexing.vector_index", fresh)
    monkeypatch.setattr("rag.dedup.semantic.vector_index", fresh)
    monkeypatch.setattr("rag.ingestion.pipeline.vector_index", fresh)
    monkeypatch.setattr("rag.retrieval.vector_index", fresh)
    yield fresh


@pytest.fixture(autouse=True)
def fresh_near_dedup_index(monkeypatch):
    """Same idea for near_dedup's in-memory MinHash/LSH index."""
    from datasketch import MinHashLSH
    from config import NEAR_DUP_JACCARD_THRESHOLD, MINHASH_NUM_PERM
    fresh_lsh = MinHashLSH(threshold=NEAR_DUP_JACCARD_THRESHOLD, num_perm=MINHASH_NUM_PERM)
    monkeypatch.setattr("rag.dedup.near._lsh", fresh_lsh)
    yield fresh_lsh


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Deterministic, instant fake embeddings -- avoids downloading/running
    a real model in unit tests. Same text always -> same vector, so exact
    and semantic dedup logic still behaves predictably. NOT meant to carry
    real semantic meaning -- see test_retrieval.py for what that limits."""
    import hashlib

    def _embed(texts):
        vectors = []
        for t in texts:
            h = hashlib.md5(t.encode()).digest()
            raw = (h * 25)[:384]
            v = np.frombuffer(bytes(raw), dtype=np.uint8).astype("float32")
            v = v / (np.linalg.norm(v) + 1e-8)
            vectors.append(v)
        return np.array(vectors, dtype="float32")

    def _embed_query(text):
        return _embed([text])

    monkeypatch.setattr("rag.embeddings.embed_texts", _embed)
    monkeypatch.setattr("rag.embeddings.embed_query", _embed_query)
    # Patch where used, per the note above:
    monkeypatch.setattr("rag.ingestion.pipeline.embed_texts", _embed)
    monkeypatch.setattr("rag.retrieval.embed_query", _embed_query)
    yield _embed