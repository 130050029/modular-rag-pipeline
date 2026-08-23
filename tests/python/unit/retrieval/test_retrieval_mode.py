import config as config
from rag.retrieval import retrieval


def _mock_chunk():
    return {
        "parent_chunk_id": "parent-1",
        "content": "test result",
        "source": "test.txt",
    }


def test_dense_mode(monkeypatch):
    monkeypatch.setattr(config, "SEARCH_MODE", "dense")

    monkeypatch.setattr(
        retrieval,
        "embed_query",
        lambda query: [0.1, 0.2],
    )

    monkeypatch.setattr(
        retrieval.vector_index,
        "search",
        lambda vector, top_k: [("chunk-1", 0.9)],
    )

    monkeypatch.setattr(
        retrieval,
        "get_chunks_with_parent_by_ids",
        lambda ids: {"chunk-1": _mock_chunk()},
    )

    results = retrieval.retrieve("test query", top_k=1)

    assert len(results) == 1
    assert results[0]["source"] == "test.txt"
    assert results[0]["score_type"] == "dense"


def test_sparse_mode(monkeypatch):
    monkeypatch.setattr(config, "SEARCH_MODE", "sparse")

    monkeypatch.setattr(
        retrieval,
        "keyword_search",
        lambda query, top_k: [("chunk-1", -2.5)],
    )

    monkeypatch.setattr(
        retrieval,
        "get_chunks_with_parent_by_ids",
        lambda ids: {"chunk-1": _mock_chunk()},
    )

    results = retrieval.retrieve("test query", top_k=1)

    assert len(results) == 1
    assert results[0]["source"] == "test.txt"
    assert results[0]["score_type"] == "sparse"


def test_hybrid_mode(monkeypatch):
    monkeypatch.setattr(config, "SEARCH_MODE", "hybrid")

    monkeypatch.setattr(
        retrieval,
        "embed_query",
        lambda query: [0.1, 0.2],
    )

    monkeypatch.setattr(
        retrieval.vector_index,
        "search",
        lambda vector, top_k: [("chunk-1", 0.9)],
    )

    monkeypatch.setattr(
        retrieval,
        "keyword_search",
        lambda query, top_k: [("chunk-1", -2.5)],
    )

    monkeypatch.setattr(
        retrieval,
        "reciprocal_rank_fusion",
        lambda dense, sparse, k, top_k: [("chunk-1", 0.03)],
    )

    monkeypatch.setattr(
        retrieval,
        "get_chunks_with_parent_by_ids",
        lambda ids: {"chunk-1": _mock_chunk()},
    )

    results = retrieval.retrieve("test query", top_k=1)

    assert len(results) == 1
    assert results[0]["source"] == "test.txt"
    assert results[0]["score_type"] == "rrf"


def test_reranking_disabled_by_default(monkeypatch):
    monkeypatch.setattr(config, "SEARCH_MODE", "dense")
    monkeypatch.setattr(config, "RERANK_ENABLED", False)

    monkeypatch.setattr(
        retrieval,
        "embed_query",
        lambda query: [0.1, 0.2],
    )

    monkeypatch.setattr(
        retrieval.vector_index,
        "search",
        lambda vector, top_k: [("chunk-1", 0.9)],
    )

    monkeypatch.setattr(
        retrieval,
        "get_chunks_with_parent_by_ids",
        lambda ids: {"chunk-1": _mock_chunk()},
    )

    results = retrieval.retrieve("test query", top_k=1)

    assert results[0]["score_type"] == "dense"


def test_reranking_uses_extra_candidates(monkeypatch):
    monkeypatch.setattr(config, "SEARCH_MODE", "dense")
    monkeypatch.setattr(config, "RERANK_ENABLED", True)
    monkeypatch.setattr(config, "RERANK_CANDIDATE_K", 5)

    seen = {}

    monkeypatch.setattr(
        retrieval,
        "embed_query",
        lambda query: [0.1, 0.2],
    )

    def fake_search(vector, top_k):
        seen["top_k"] = top_k
        return [("chunk-1", 0.9)]

    monkeypatch.setattr(
        retrieval.vector_index,
        "search",
        fake_search,
    )

    monkeypatch.setattr(
        retrieval,
        "get_chunks_with_parent_by_ids",
        lambda ids: {"chunk-1": _mock_chunk()},
    )

    monkeypatch.setattr(
        retrieval,
        "rerank",
        lambda query, results, top_k: results[:top_k],
    )

    retrieval.retrieve("test query", top_k=2)

    assert seen["top_k"] == 5