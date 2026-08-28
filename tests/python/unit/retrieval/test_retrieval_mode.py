import pytest
import config

from rag.retrieval import retrieval


def _mock_chunk(chunk_id="chunk-1"):
    return {
        "parent_chunk_id": "parent-1",
        "content": "test result",
        "source": "test.txt",
    }


def _patch_chunk_lookup(monkeypatch, chunk_id="chunk-1"):
    monkeypatch.setattr(
        retrieval,
        "get_chunks_with_parent_by_ids",
        lambda ids: {
            chunk_id: _mock_chunk(chunk_id),
        },
    )


def test_dense_mode_calls_vector_search(monkeypatch):
    monkeypatch.setattr(config, "SEARCH_MODE", "dense")
    monkeypatch.setattr(config, "RERANK_ENABLED", False)

    calls = {}

    monkeypatch.setattr(
        retrieval,
        "embed_query",
        lambda query: [0.1, 0.2],
    )

    def fake_search(vector, top_k):
        calls["top_k"] = top_k
        return [("chunk-1", 0.9)]

    monkeypatch.setattr(
        retrieval.vector_index,
        "search",
        fake_search,
    )

    _patch_chunk_lookup(monkeypatch)

    results = retrieval.retrieve(
        "test query",
        top_k=3,
    )

    assert calls["top_k"] == 3
    assert results[0]["score_type"] == "dense"


def test_sparse_mode_calls_keyword_search(monkeypatch):
    monkeypatch.setattr(config, "SEARCH_MODE", "sparse")
    monkeypatch.setattr(config, "RERANK_ENABLED", False)

    calls = {}

    def fake_keyword_search(query, top_k):
        calls["top_k"] = top_k
        return [("chunk-1", -2.5)]

    monkeypatch.setattr(
        retrieval,
        "keyword_search",
        fake_keyword_search,
    )

    _patch_chunk_lookup(monkeypatch)

    results = retrieval.retrieve(
        "test query",
        top_k=3,
    )

    assert calls["top_k"] == 3
    assert results[0]["score_type"] == "sparse"


def test_hybrid_mode_calls_both_retrievers(monkeypatch):
    monkeypatch.setattr(config, "SEARCH_MODE", "hybrid")
    monkeypatch.setattr(config, "RERANK_ENABLED", False)
    monkeypatch.setattr(config, "DENSE_CANDIDATE_K", 7)
    monkeypatch.setattr(config, "SPARSE_CANDIDATE_K", 9)

    calls = {}

    monkeypatch.setattr(
        retrieval,
        "embed_query",
        lambda query: [0.1, 0.2],
    )

    def fake_dense(vector, top_k):
        calls["dense_top_k"] = top_k
        return [("chunk-1", 0.9)]

    def fake_sparse(query, top_k):
        calls["sparse_top_k"] = top_k
        return [("chunk-1", -2.5)]

    monkeypatch.setattr(
        retrieval.vector_index,
        "search",
        fake_dense,
    )

    monkeypatch.setattr(
        retrieval,
        "keyword_search",
        fake_sparse,
    )

    def fake_rrf(dense, sparse, k, top_k):
        calls["rrf"] = (dense, sparse, k, top_k)
        return [("chunk-1", 0.03)]

    monkeypatch.setattr(
        retrieval,
        "reciprocal_rank_fusion",
        fake_rrf,
    )

    _patch_chunk_lookup(monkeypatch)

    results = retrieval.retrieve(
        "test query",
        top_k=2,
    )

    assert calls["dense_top_k"] == 7
    assert calls["sparse_top_k"] == 9
    assert calls["rrf"][3] == 2
    assert results[0]["score_type"] == "rrf"


def test_reranking_is_disabled_without_rerank_call(monkeypatch):
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

    _patch_chunk_lookup(monkeypatch)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("rerank() must not be called")

    monkeypatch.setattr(
        retrieval,
        "rerank",
        fail_if_called,
    )

    results = retrieval.retrieve(
        "test query",
        top_k=1,
    )

    assert results[0]["score_type"] == "dense"


def test_reranking_receives_extra_candidates(monkeypatch):
    monkeypatch.setattr(config, "SEARCH_MODE", "dense")
    monkeypatch.setattr(config, "RERANK_ENABLED", True)
    monkeypatch.setattr(config, "RERANK_CANDIDATE_K", 5)

    calls = {}

    monkeypatch.setattr(
        retrieval,
        "embed_query",
        lambda query: [0.1, 0.2],
    )

    def fake_search(vector, top_k):
        calls["top_k"] = top_k
        return [("chunk-1", 0.9)]

    monkeypatch.setattr(
        retrieval.vector_index,
        "search",
        fake_search,
    )

    _patch_chunk_lookup(monkeypatch)

    monkeypatch.setattr(
        retrieval,
        "rerank",
        lambda query, results, top_k: results[:top_k],
    )

    retrieval.retrieve(
        "test query",
        top_k=2,
    )

    assert calls["top_k"] == 5


def test_non_positive_top_k_returns_empty_without_search(monkeypatch):
    monkeypatch.setattr(config, "SEARCH_MODE", "dense")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("retriever should not be called")

    monkeypatch.setattr(
        retrieval.vector_index,
        "search",
        fail_if_called,
    )

    assert retrieval.retrieve("query", top_k=0) == []
    assert retrieval.retrieve("query", top_k=-1) == []


@pytest.mark.parametrize(
    "mode",
    [
        "invalid",
        "bm25",
        "",
    ],
)
def test_invalid_search_mode_is_rejected(monkeypatch, mode):
    monkeypatch.setattr(config, "SEARCH_MODE", mode)

    with pytest.raises(ValueError, match="Unsupported SEARCH_MODE"):
        retrieval.retrieve("query", top_k=1)