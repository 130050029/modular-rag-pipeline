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

import sys
from pathlib import Path

# Ensure the project root (parent of tests/) is importable regardless of
# where pytest is invoked from -- MUST happen before `import config` below,
# since config.py lives at the project root, not inside tests/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Importing config here (before anything that might pull in faiss/torch)
# is what applies its centralized KMP_DUPLICATE_LIB_OK/OMP_NUM_THREADS fix --
# pytest imports every test file during collection, and whichever one
# happens to import rag.embeddings first will pull in torch; if that
# happens before this import runs, FAISS's search() can segfault. See
# config.py's own comment for the full explanation.
import config  # noqa: F401  (imported for its module-level side effect)

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Point the database at a fresh temp SQLite file for every test.

    DB_PATH is still patched on rag.storage.connection (it stays a plain
    module-level constant there). DB_BACKEND, however, is patched on
    config.DB_BACKEND directly, NOT rag.storage.connection.DB_BACKEND --
    connection.py no longer holds DB_BACKEND as a module-level attribute at
    all (Connection.__init__ re-imports it fresh from config on every call,
    specifically so a monkeypatched value here is always picked up
    regardless of what's set in the real shell environment). Patching the
    old location would now raise AttributeError instead of silently doing
    nothing -- this is the corrected version of that fixture."""
    db_file = tmp_path / "test_rag.db"
    monkeypatch.setattr("rag.storage.connection.DB_PATH", str(db_file))
    monkeypatch.setattr("config.DB_BACKEND", "sqlite")

    # Also isolate FAISS index persistence to this test's tmp_path --
    # without this, test_api.py's `with TestClient(server.app) as c:` would
    # trigger lifespan's real load_from_disk()/save() calls against
    # config.FAISS_INDEX_PATH's REAL value ("data/faiss_index"), silently
    # reading and writing actual persisted index files on disk during tests.
    faiss_prefix = tmp_path / "test_faiss_index"
    monkeypatch.setattr("config.FAISS_INDEX_PATH", str(faiss_prefix))

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
    """Fresh in-memory MinHash/LSH backend for every test, and forces
    config.NEAR_DUP_BACKEND to "memory" regardless of the real
    environment's setting -- same reasoning as temp_db forcing DB_BACKEND
    to "sqlite": avoids the exact class of bug hit earlier, where a shell
    env var leaking into the test process caused the wrong backend to be
    selected and a required import to be skipped."""
    import rag.dedup.near as near_module
    fresh_backend = near_module._MemoryBackend()
    monkeypatch.setattr(near_module, "_memory_backend", fresh_backend)
    monkeypatch.setattr("config.NEAR_DUP_BACKEND", "memory")
    yield fresh_backend


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