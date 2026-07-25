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


def test_save_and_load_round_trip(tmp_path):
    """The persistence mechanism lifespan relies on to avoid re-embedding
    the entire corpus on every restart -- verify save() + load_from_disk()
    actually preserves searchable state, including a removal that happened
    before saving."""
    idx1 = VectorIndex()
    v1, v2, v3 = _unit_vector(1), _unit_vector(2), _unit_vector(3)
    idx1.add(np.array([v1]), ["chunk-a"])
    idx1.add(np.array([v2]), ["chunk-b"])
    idx1.add(np.array([v3]), ["chunk-c"])
    idx1.remove(["chunk-b"])   # exercise removal before saving

    path_prefix = str(tmp_path / "test_index")
    idx1.save(path_prefix)

    idx2 = VectorIndex()   # fresh, empty -- simulating a brand new process
    assert idx2.size == 0

    loaded = idx2.load_from_disk(path_prefix)
    assert loaded is True
    assert idx2.size == 2   # chunk-b was removed before saving, shouldn't reappear

    results = idx2.search(np.array([v1]), top_k=3)
    result_ids = [r[0] for r in results]
    assert "chunk-a" in result_ids
    assert "chunk-c" in result_ids
    assert "chunk-b" not in result_ids


def test_load_from_disk_returns_false_when_nothing_persisted(tmp_path):
    idx = VectorIndex()
    loaded = idx.load_from_disk(str(tmp_path / "does_not_exist"))
    assert loaded is False
    assert idx.size == 0


def test_can_continue_adding_after_load_without_id_collision(tmp_path):
    """Confirms _next_id is correctly restored -- a loaded index must be
    able to accept new vectors without clashing with IDs already in use."""
    idx1 = VectorIndex()
    idx1.add(np.array([_unit_vector(1)]), ["chunk-a"])
    path_prefix = str(tmp_path / "test_index")
    idx1.save(path_prefix)

    idx2 = VectorIndex()
    idx2.load_from_disk(path_prefix)
    idx2.add(np.array([_unit_vector(4)]), ["chunk-d"])

    assert idx2.size == 2
    results = idx2.search(np.array([_unit_vector(4)]), top_k=1)
    assert results[0][0] == "chunk-d"
