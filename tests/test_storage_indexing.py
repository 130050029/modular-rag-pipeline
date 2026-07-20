import numpy as np
from rag.storage.indexing import VectorIndex


def _unit_vector(seed):
    rng = np.random.RandomState(seed)
    v = rng.rand(384).astype("float32")
    return v / np.linalg.norm(v)


def test_add_and_search_returns_expected_chunk():
    idx = VectorIndex()
    v1, v2 = _unit_vector(1), _unit_vector(2)
    idx.add(np.array([v1]), ["chunk-a"])
    idx.add(np.array([v2]), ["chunk-b"])

    results = idx.search(np.array([v1]), top_k=1)
    assert results[0][0] == "chunk-a"
    assert idx.size == 2


def test_remove_actually_excludes_from_search():
    idx = VectorIndex()
    v1 = _unit_vector(1)
    idx.add(np.array([v1]), ["chunk-a"])
    idx.remove(["chunk-a"])

    assert idx.size == 0
    assert idx.search(np.array([v1]), top_k=1) == []
