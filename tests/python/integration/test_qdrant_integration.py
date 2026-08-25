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

        client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            timeout=2,
        )

        client.get_collections()
        return True

    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _qdrant_reachable(),
    reason=(
        "Qdrant not reachable at config.py's configured host/port -- "
        "run `docker compose up -d` to enable these tests."
    ),
)


@pytest.fixture
def qdrant_backend(monkeypatch):
    monkeypatch.setattr(
        "config.VECTOR_BACKEND",
        "qdrant",
    )

    monkeypatch.setattr(
        "config.QDRANT_COLLECTION",
        TEST_COLLECTION,
    )

    from qdrant_client import QdrantClient
    from config import QDRANT_HOST, QDRANT_PORT

    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )

    if client.collection_exists(TEST_COLLECTION):
        client.delete_collection(TEST_COLLECTION)

    from rag.storage.qdrant_index import QdrantVectorIndex

    idx = QdrantVectorIndex()

    yield idx

    if client.collection_exists(TEST_COLLECTION):
        client.delete_collection(TEST_COLLECTION)


def test_add_and_search_returns_expected_chunk(qdrant_backend):
    chunk_a = str(uuid.uuid4())
    chunk_b = str(uuid.uuid4())

    v1 = _unit_vector(1)
    v2 = _unit_vector(2)

    qdrant_backend.add(
        np.array([v1]),
        [chunk_a],
    )

    qdrant_backend.add(
        np.array([v2]),
        [chunk_b],
    )

    results = qdrant_backend.search(
        np.array([v1]),
        top_k=1,
    )

    assert results
    assert results[0][0] == chunk_a
    assert qdrant_backend.size == 2


def test_search_returns_multiple_ranked_results(qdrant_backend):
    """The adapter must expose Qdrant's ranked results through the same
    (chunk_id, score) interface used by FAISS."""
    chunk_a = str(uuid.uuid4())
    chunk_b = str(uuid.uuid4())

    v1 = _unit_vector(1)
    v2 = _unit_vector(2)

    qdrant_backend.add(
        np.array([v1, v2]),
        [chunk_a, chunk_b],
    )

    results = qdrant_backend.search(
        np.array([v1]),
        top_k=2,
    )

    result_ids = [
        chunk_id
        for chunk_id, _score in results
    ]

    assert chunk_a in result_ids
    assert chunk_b in result_ids

    scores = [
        score
        for _chunk_id, score in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )


def test_remove_is_true_deletion_not_tombstone(qdrant_backend):
    """Unlike our FAISS/HNSW wrapper, Qdrant genuinely deletes -- size
    should drop for real, not just diverge from an underlying raw count."""
    chunk_a = str(uuid.uuid4())

    v1 = _unit_vector(1)

    qdrant_backend.add(
        np.array([v1]),
        [chunk_a],
    )

    assert qdrant_backend.size == 1

    qdrant_backend.remove([chunk_a])

    assert qdrant_backend.size == 0
    assert qdrant_backend.search(
        np.array([v1]),
        top_k=1,
    ) == []


def test_remove_one_chunk_preserves_other_chunks(qdrant_backend):
    chunk_a = str(uuid.uuid4())
    chunk_b = str(uuid.uuid4())

    v1 = _unit_vector(1)
    v2 = _unit_vector(2)

    qdrant_backend.add(
        np.array([v1, v2]),
        [chunk_a, chunk_b],
    )

    assert qdrant_backend.size == 2

    qdrant_backend.remove([chunk_a])

    assert qdrant_backend.size == 1

    results = qdrant_backend.search(
        np.array([v2]),
        top_k=10,
    )

    result_ids = [
        chunk_id
        for chunk_id, _score in results
    ]

    assert chunk_b in result_ids
    assert chunk_a not in result_ids


def test_upsert_existing_chunk_id_replaces_vector(qdrant_backend):
    """Qdrant uses point IDs as stable chunk IDs. Adding the same chunk ID
    again therefore behaves as an upsert rather than creating a second point.
    """
    chunk_id = str(uuid.uuid4())

    v1 = _unit_vector(1)
    v2 = _unit_vector(2)

    qdrant_backend.add(
        np.array([v1]),
        [chunk_id],
    )

    assert qdrant_backend.size == 1

    qdrant_backend.add(
        np.array([v2]),
        [chunk_id],
    )

    assert qdrant_backend.size == 1

    results = qdrant_backend.search(
        np.array([v2]),
        top_k=1,
    )

    assert results
    assert results[0][0] == chunk_id


def test_save_and_load_are_safe_no_ops(qdrant_backend):
    """save()/load_from_disk() should never error, and load_from_disk()
    should always report success (True) -- Qdrant persists on its own, so
    there's genuinely nothing for our code to do here."""
    qdrant_backend.save(
        "irrelevant_path"
    )

    assert (
        qdrant_backend.load_from_disk(
            "irrelevant_path"
        )
        is True
    )


def test_data_survives_new_qdrant_backend_instance(qdrant_backend):
    """A new adapter instance pointed at the same collection must see data
    already persisted in Qdrant. This is the distributed-storage equivalent
    of the FAISS save/load test."""
    chunk_id = str(uuid.uuid4())
    vector = _unit_vector(7)

    qdrant_backend.add(
        np.array([vector]),
        [chunk_id],
    )

    assert qdrant_backend.size == 1

    from rag.storage.qdrant_index import QdrantVectorIndex

    fresh_idx = QdrantVectorIndex()

    results = fresh_idx.search(
        np.array([vector]),
        top_k=1,
    )

    assert results
    assert results[0][0] == chunk_id

    fresh_idx.remove([chunk_id])