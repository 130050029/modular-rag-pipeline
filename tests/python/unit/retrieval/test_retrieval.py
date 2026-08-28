import config

from rag.ingestion.pipeline import ingest_document
from rag.retrieval.retrieval import retrieve


def test_retrieve_returns_parent_content(fake_embeddings):
    ingest_document(
        "doc.txt",
        "Paris is the capital of France and a major European city "
        "with many landmarks.",
    )

    results = retrieve(
        "What is the capital of France?",
        top_k=1,
    )

    assert len(results) == 1
    assert "Paris" in results[0]["content"]
    assert results[0]["source"] == "doc.txt"


def test_retrieve_returns_expected_result_metadata(fake_embeddings):
    ingest_document(
        "doc.txt",
        "Paris is the capital of France.",
    )

    results = retrieve(
        "capital of France",
        top_k=1,
    )

    result = results[0]

    assert result["chunk_id"]
    assert result["parent_chunk_id"]
    assert result["source"] == "doc.txt"
    assert isinstance(result["score"], float)
    assert result["score_type"] in {
        "dense",
        "sparse",
        "rrf",
    }
    assert result["rank"] == 1


def test_retrieve_with_empty_index_returns_nothing(
    fake_embeddings,
):
    assert retrieve("anything at all") == []


def test_dense_retrieval_returns_top_k(
    fake_embeddings,
    monkeypatch,
):
    monkeypatch.setattr(
        config,
        "SEARCH_MODE",
        "dense",
    )

    ingest_document(
        "doc-a.txt",
        "Paris is the capital of France.",
    )

    ingest_document(
        "doc-b.txt",
        "Berlin is the capital of Germany.",
    )

    results = retrieve(
        "capital",
        top_k=1,
    )

    assert len(results) == 1


def test_sparse_retrieval_finds_lexically_matching_content(
    fake_embeddings,
    monkeypatch,
):
    monkeypatch.setattr(
        config,
        "SEARCH_MODE",
        "sparse",
    )

    ingest_document(
        "policy.txt",
        "Remote employees may work from home four days per week.",
    )

    ingest_document(
        "unrelated.txt",
        "The company provides health insurance and retirement benefits.",
    )

    results = retrieve(
        "four days per week",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["source"] == "policy.txt"
    assert results[0]["score_type"] == "sparse"


def test_hybrid_retrieval_combines_dense_and_sparse(
    fake_embeddings,
    monkeypatch,
):
    monkeypatch.setattr(
        config,
        "SEARCH_MODE",
        "hybrid",
    )

    ingest_document(
        "policy.txt",
        "Remote employees may work from home four days per week.",
    )

    results = retrieve(
        "remote work four days",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["source"] == "policy.txt"
    assert results[0]["score_type"] == "rrf"


def test_retrieve_ignores_missing_database_rows(
    monkeypatch,
):
    monkeypatch.setattr(
        config,
        "SEARCH_MODE",
        "dense",
    )

    monkeypatch.setattr(
        "rag.retrieval.retrieval.embed_query",
        lambda query: [0.1, 0.2],
    )

    monkeypatch.setattr(
        "rag.retrieval.retrieval.vector_index.search",
        lambda vector, top_k: [
            ("missing-chunk", 0.99),
            ("existing-chunk", 0.80),
        ],
    )

    monkeypatch.setattr(
        "rag.retrieval.retrieval.get_chunks_with_parent_by_ids",
        lambda ids: {
            "existing-chunk": {
                "parent_chunk_id": "parent-1",
                "content": "Existing content",
                "source": "existing.txt",
            }
        },
    )

    results = retrieve(
        "test query",
        top_k=2,
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "existing-chunk"


def test_retrieve_applies_reranking_when_enabled(
    monkeypatch,
):
    monkeypatch.setattr(
        config,
        "SEARCH_MODE",
        "dense",
    )

    monkeypatch.setattr(
        config,
        "RERANK_ENABLED",
        True,
    )

    monkeypatch.setattr(
        config,
        "RERANK_CANDIDATE_K",
        3,
    )

    monkeypatch.setattr(
        "rag.retrieval.retrieval.embed_query",
        lambda query: [0.1, 0.2],
    )

    monkeypatch.setattr(
        "rag.retrieval.retrieval.vector_index.search",
        lambda vector, top_k: [
            ("chunk-a", 0.9),
            ("chunk-b", 0.8),
        ],
    )

    monkeypatch.setattr(
        "rag.retrieval.retrieval.get_chunks_with_parent_by_ids",
        lambda ids: {
            "chunk-a": {
                "parent_chunk_id": "parent-a",
                "content": "A",
                "source": "a.txt",
            },
            "chunk-b": {
                "parent_chunk_id": "parent-b",
                "content": "B",
                "source": "b.txt",
            },
        },
    )

    def fake_rerank(query, results, top_k):
        assert query == "test query"
        assert len(results) == 2
        assert top_k == 1

        result = dict(results[1])
        result["score"] = 0.99
        result["score_type"] = "reranker"
        result["retrieval_score"] = results[1]["score"]
        result["rank"] = 1

        return [result]

    monkeypatch.setattr(
        "rag.retrieval.retrieval.rerank",
        fake_rerank,
    )

    results = retrieve(
        "test query",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["source"] == "b.txt"
    assert results[0]["score_type"] == "reranker"
    assert results[0]["rank"] == 1