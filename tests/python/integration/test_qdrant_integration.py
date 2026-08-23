"""
test_qdrant_integration.py -- exercises the actual Qdrant-backed vector
index for real, the same way test_postgres_integration.py and
test_redis_integration.py exercise their respective backends.

Uses a dedicated test collection name (not the real configured
QDRANT_COLLECTION), deleted before and after each test -- mirroring
test_redis_integration.py's dedicated-DB-15 pattern -- so this never
touches real data even if pointed at a shared Qdrant instance.

Requires a running Qdrant matching config.py's defaults, e.g.:
    docker compose up -d

If Qdrant isn't reachable, these tests SKIP (not fail).
"""

import uuid
import numpy as np
import pytest

TEST_COLLECTION = "rag_chunks_test"


def _unit_vector(seed):
    rng = np.random.RandomState(seed)
    v = rng.rand(384).astype("float32")
    return v / np.linalg.norm(v)


def _qdrant_reachable() -> bool:
    try:
        from qdrant_client import QdrantClient
        from config import QDRANT_HOST, QDRANT_PORT
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
        client.get_collections()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _qdrant_reachable(),
    reason="Qdrant not reachable at config.py's configured host/port -- "
           "run `docker compose up -d` to enable these tests.",
)


@pytest.fixture
def qdrant_backend(monkeypatch):
    monkeypatch.setattr("config.VECTOR_BACKEND", "qdrant")
    monkeypatch.setattr("config.QDRANT_COLLECTION", TEST_COLLECTION)

    from qdrant_client import QdrantClient
    from config import QDRANT_HOST, QDRANT_PORT
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    if client.collection_exists(TEST_COLLECTION):
        client.delete_collection(TEST_COLLECTION)

    from rag.storage.qdrant_index import QdrantVectorIndex
    idx = QdrantVectorIndex()
    yield idx

    if client.collection_exists(TEST_COLLECTION):
        client.delete_collection(TEST_COLLECTION)


def test_add_and_search_returns_expected_chunk(qdrant_backend):
    chunk_a, chunk_b = str(uuid.uuid4()), str(uuid.uuid4())
    v1, v2 = _unit_vector(1), _unit_vector(2)

    qdrant_backend.add(np.array([v1]), [chunk_a])
    qdrant_backend.add(np.array([v2]), [chunk_b])

    results = qdrant_backend.search(np.array([v1]), top_k=1)
    assert results[0][0] == chunk_a
    assert qdrant_backend.size == 2


def test_remove_is_true_deletion_not_tombstone(qdrant_backend):
    """Unlike our FAISS/HNSW wrapper, Qdrant genuinely deletes -- size
    should drop for real, not just diverge from an underlying raw count."""
    chunk_a = str(uuid.uuid4())
    v1 = _unit_vector(1)

    qdrant_backend.add(np.array([v1]), [chunk_a])
    assert qdrant_backend.size == 1

    qdrant_backend.remove([chunk_a])
    assert qdrant_backend.size == 0
    assert qdrant_backend.search(np.array([v1]), top_k=1) == []


def test_save_and_load_are_safe_no_ops(qdrant_backend):
    """save()/load_from_disk() should never error, and load_from_disk()
    should always report success (True) -- Qdrant persists on its own, so
    there's genuinely nothing for our code to do here."""
    qdrant_backend.save("irrelevant_path")   # should not raise
    assert qdrant_backend.load_from_disk("irrelevant_path") is True
