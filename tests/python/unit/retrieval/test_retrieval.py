import config
import pytest

from rag.ingestion.pipeline import ingest_document
from rag.retrieval import retrieval
from rag.query.processing import DefaultQueryProcessor

def test_retrieve_returns_parent_content(fake_embeddings):
    ingest_document(
        "doc.txt",
        "Paris is the capital of France and a major European city "
        "with many landmarks.",
    )

    results =retrieval.retrieve(
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

    results =retrieval.retrieve(
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
    assert retrieval.retrieve("anything at all") == []


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

    results =retrieval.retrieve(
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

    results =retrieval.retrieve(
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

    results =retrieval.retrieve(
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

    results =retrieval.retrieve(
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

    results =retrieval.retrieve(
        "test query",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["source"] == "b.txt"
    assert results[0]["score_type"] == "reranker"
    assert results[0]["rank"] == 1

def test_retrieval_uses_processed_query(monkeypatch):
    monkeypatch.setattr(config, "SEARCH_MODE", "sparse")
    monkeypatch.setattr(config, "RERANK_ENABLED", False)

    captured = {}

    class FakeProcessor:
        def process(self, query):
            captured["original"] = query
            return ["rewritten retrieval query"]

    monkeypatch.setattr(
        "rag.retrieval.retrieval.get_query_processor",
        lambda: FakeProcessor(),
    )

    def fake_keyword_search(query, top_k):
        captured["retrieval_query"] = query
        return [("chunk-1", -1.0)]

    monkeypatch.setattr(
        "rag.retrieval.retrieval.keyword_search",
        fake_keyword_search,
    )

    monkeypatch.setattr(
        "rag.retrieval.retrieval.get_chunks_with_parent_by_ids",
        lambda ids: {
            "chunk-1": {
                "parent_chunk_id": "parent-1",
                "content": "Relevant content",
                "source": "doc.txt",
            }
        },
    )

    results =retrieval.retrieve(
        "original user question",
        top_k=1,
    )

    assert results[0]["source"] == "doc.txt"
    assert captured["original"] == "original user question"
    assert captured["retrieval_query"] == "rewritten retrieval query"

def test_retrieve_runs_each_processed_query(monkeypatch):
    monkeypatch.setattr(config, "SEARCH_MODE", "sparse")
    monkeypatch.setattr(config, "RERANK_ENABLED", False)

    class FakeProcessor:
        def process(self, query):
            return ["query one", "query two"]

    monkeypatch.setattr(
        retrieval,
        "get_query_processor",
        lambda: FakeProcessor(),
    )

    calls = []

    def fake_keyword_search(query, top_k):
        calls.append(query)

        if query == "query one":
            return [("chunk-1", -1.0)]

        if query == "query two":
            return [("chunk-2", -0.5)]

        raise AssertionError(f"Unexpected query: {query}")

    monkeypatch.setattr(
        retrieval,
        "keyword_search",
        fake_keyword_search,
    )

    def fake_chunk_lookup(ids):
        return {
            "chunk-1": {
                "parent_chunk_id": "parent-1",
                "content": "Content one",
                "source": "one.txt",
            },
            "chunk-2": {
                "parent_chunk_id": "parent-2",
                "content": "Content two",
                "source": "two.txt",
            },
        }

    monkeypatch.setattr(
        retrieval,
        "get_chunks_with_parent_by_ids",
        fake_chunk_lookup,
    )

    monkeypatch.setattr(
        retrieval,
        "multi_query_fusion",
        lambda query_results, k, top_k: [
            ("chunk-1", 0.02),
        ],
    )

    results = retrieval.retrieve(
        "original question",
        top_k=1,
    )

    assert calls == ["query one", "query two"]
    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk-1"
    assert results[0]["score_type"] == "multi_query"

def test_retrieve_single_processed_query_keeps_normal_score_type(
    monkeypatch,
):
    monkeypatch.setattr(config, "SEARCH_MODE", "sparse")
    monkeypatch.setattr(config, "RERANK_ENABLED", False)

    class FakeProcessor:
        def process(self, query):
            return ["processed query"]

    monkeypatch.setattr(
        retrieval,
        "get_query_processor",
        lambda: FakeProcessor(),
    )

    monkeypatch.setattr(
        retrieval,
        "keyword_search",
        lambda query, top_k: [
            ("chunk-1", -1.0),
        ],
    )

    monkeypatch.setattr(
        retrieval,
        "get_chunks_with_parent_by_ids",
        lambda ids: {
            "chunk-1": {
                "parent_chunk_id": "parent-1",
                "content": "Relevant content",
                "source": "doc.txt",
            }
        },
    )

    results = retrieval.retrieve(
        "original question",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["source"] == "doc.txt"
    assert results[0]["score_type"] == "sparse"

def test_retrieve_fuses_multi_query_results(monkeypatch):
    monkeypatch.setattr(config, "SEARCH_MODE", "sparse")
    monkeypatch.setattr(config, "RERANK_ENABLED", False)

    class FakeProcessor:
        def process(self, query):
            return ["query one", "query two", "query three"]

    monkeypatch.setattr(
        retrieval,
        "get_query_processor",
        lambda: FakeProcessor(),
    )

    monkeypatch.setattr(
        retrieval,
        "keyword_search",
        lambda query, top_k: [
            (f"chunk-{query[-1]}", 1.0),
        ],
    )

    monkeypatch.setattr(
        retrieval,
        "get_chunks_with_parent_by_ids",
        lambda ids: {
            "chunk-e": {
                "parent_chunk_id": "parent-e",
                "content": "Content E",
                "source": "doc-e.txt",
            },
            "chunk-o": {
                "parent_chunk_id": "parent-o",
                "content": "Content O",
                "source": "doc-o.txt",
            },
        },
    )

    captured = {}

    def fake_fusion(query_results, k, top_k):
        captured["query_results"] = query_results
        captured["k"] = k
        captured["top_k"] = top_k

        return [("chunk-e", 0.5)]

    monkeypatch.setattr(
        retrieval,
        "multi_query_fusion",
        fake_fusion,
    )

    results = retrieval.retrieve(
        "original query",
        top_k=1,
    )

    assert len(captured["query_results"]) == 3

    assert captured["query_results"] == [
        [("chunk-e", 1.0)],
        [("chunk-o", 1.0)],
        [("chunk-e", 1.0)],
    ]

    assert captured["k"] == config.RRF_K
    assert captured["top_k"] == 1

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk-e"
    assert results[0]["source"] == "doc-e.txt"
    assert results[0]["score_type"] == "multi_query"

def test_retrieve_multi_query_marks_final_score_as_multi_query(
    monkeypatch,
):
    monkeypatch.setattr(config, "SEARCH_MODE", "sparse")
    monkeypatch.setattr(config, "RERANK_ENABLED", False)

    class FakeProcessor:
        def process(self, query):
            return ["query one", "query two"]

    monkeypatch.setattr(
        retrieval,
        "get_query_processor",
        lambda: FakeProcessor(),
    )

    monkeypatch.setattr(
        retrieval,
        "keyword_search",
        lambda query, top_k: [
            ("chunk-1", -1.0),
        ],
    )

    monkeypatch.setattr(
        retrieval,
        "get_chunks_with_parent_by_ids",
        lambda ids: {
            "chunk-1": {
                "parent_chunk_id": "parent-1",
                "content": "Relevant content",
                "source": "doc.txt",
            }
        },
    )

    monkeypatch.setattr(
        retrieval,
        "multi_query_fusion",
        lambda query_results, k, top_k: [
            ("chunk-1", 0.5),
        ],
    )

    results = retrieval.retrieve(
        "original question",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["source"] == "doc.txt"
    assert results[0]["score_type"] == "multi_query"

# ---------------------------------------------------------------------------
# Query routing -> processing -> retrieval
# ---------------------------------------------------------------------------


def test_retrieve_uses_real_processor_routing_for_simple_query(
    monkeypatch,
):
    monkeypatch.setattr(config, "SEARCH_MODE", "sparse")
    monkeypatch.setattr(config, "RERANK_ENABLED", False)

    calls = []

    class FakeRouter:
        def is_complex(self, query):
            calls.append(("route", query))
            return False

    class FakeRewriter:
        def rewrite(self, query):
            calls.append(("rewrite", query))
            return "rewritten retrieval query"

    class FakeDecomposer:
        def decompose(self, query):
            calls.append(("decompose", query))
            raise AssertionError(
                "Decomposer must not run for a simple query"
            )

    processor = DefaultQueryProcessor(
        complexity_router=FakeRouter(),
        rewriter=FakeRewriter(),
        decomposer=FakeDecomposer(),
    )

    monkeypatch.setattr(
        retrieval,
        "get_query_processor",
        lambda: processor,
    )

    monkeypatch.setattr(
        retrieval,
        "keyword_search",
        lambda query, top_k: [
            ("chunk-1", -1.0),
        ],
    )

    monkeypatch.setattr(
        retrieval,
        "get_chunks_with_parent_by_ids",
        lambda ids: {
            "chunk-1": {
                "parent_chunk_id": "parent-1",
                "content": "Relevant content",
                "source": "doc.txt",
            }
        },
    )

    results = retrieval.retrieve(
        "original user question",
        top_k=1,
    )

    assert calls == [
        ("route", "original user question"),
        ("rewrite", "original user question"),
    ]

    assert len(results) == 1
    assert results[0]["source"] == "doc.txt"


def test_retrieve_uses_real_processor_routing_for_complex_query(
    monkeypatch,
):
    monkeypatch.setattr(config, "SEARCH_MODE", "sparse")
    monkeypatch.setattr(config, "RERANK_ENABLED", False)

    calls = []

    class FakeRouter:
        def is_complex(self, query):
            calls.append(("route", query))
            return True

    class FakeRewriter:
        def rewrite(self, query):
            calls.append(("rewrite", query))
            raise AssertionError(
                "Rewriter must not run for a complex query"
            )

    class FakeDecomposer:
        def decompose(self, query):
            calls.append(("decompose", query))
            return [
                "sub-question one",
                "sub-question two",
            ]

    processor = DefaultQueryProcessor(
        complexity_router=FakeRouter(),
        rewriter=FakeRewriter(),
        decomposer=FakeDecomposer(),
    )

    monkeypatch.setattr(
        retrieval,
        "get_query_processor",
        lambda: processor,
    )

    retrieval_queries = []

    def fake_keyword_search(query, top_k):
        retrieval_queries.append(query)

        return [
            ("chunk-1", -1.0),
        ]

    monkeypatch.setattr(
        retrieval,
        "keyword_search",
        fake_keyword_search,
    )

    monkeypatch.setattr(
        retrieval,
        "get_chunks_with_parent_by_ids",
        lambda ids: {
            "chunk-1": {
                "parent_chunk_id": "parent-1",
                "content": "Relevant content",
                "source": "doc.txt",
            }
        },
    )

    monkeypatch.setattr(
        retrieval,
        "multi_query_fusion",
        lambda query_results, k, top_k: [
            ("chunk-1", 0.5),
        ],
    )

    results = retrieval.retrieve(
        "complex user question",
        top_k=1,
    )

    assert calls == [
        ("route", "complex user question"),
        ("decompose", "complex user question"),
    ]

    assert retrieval_queries == [
        "sub-question one",
        "sub-question two",
    ]

    assert len(results) == 1
    assert results[0]["source"] == "doc.txt"
    assert results[0]["score_type"] == "multi_query"


def test_retrieve_preserves_expansion_after_simple_query_routing(
    monkeypatch,
):
    monkeypatch.setattr(config, "SEARCH_MODE", "sparse")
    monkeypatch.setattr(config, "RERANK_ENABLED", False)

    calls = []

    class FakeRouter:
        def is_complex(self, query):
            calls.append(("route", query))
            return False

    class FakeRewriter:
        def rewrite(self, query):
            calls.append(("rewrite", query))
            return "rewritten query"

    class FakeExpander:
        def expand(self, query):
            calls.append(("expand", query))
            return [
                "expanded query one",
                "expanded query two",
            ]

    processor = DefaultQueryProcessor(
        complexity_router=FakeRouter(),
        rewriter=FakeRewriter(),
        expander=FakeExpander(),
    )

    monkeypatch.setattr(
        retrieval,
        "get_query_processor",
        lambda: processor,
    )

    retrieval_queries = []

    def fake_keyword_search(query, top_k):
        retrieval_queries.append(query)

        return [
            ("chunk-1", -1.0),
        ]

    monkeypatch.setattr(
        retrieval,
        "keyword_search",
        fake_keyword_search,
    )

    monkeypatch.setattr(
        retrieval,
        "get_chunks_with_parent_by_ids",
        lambda ids: {
            "chunk-1": {
                "parent_chunk_id": "parent-1",
                "content": "Relevant content",
                "source": "doc.txt",
            }
        },
    )

    monkeypatch.setattr(
        retrieval,
        "multi_query_fusion",
        lambda query_results, k, top_k: [
            ("chunk-1", 0.5),
        ],
    )

    results = retrieval.retrieve(
        "original question",
        top_k=1,
    )

    assert calls == [
        ("route", "original question"),
        ("rewrite", "original question"),
        ("expand", "rewritten query"),
    ]

    assert retrieval_queries == [
        "expanded query one",
        "expanded query two",
    ]

    assert len(results) == 1
    assert results[0]["score_type"] == "multi_query"


def test_retrieve_preserves_expansion_after_complex_query_routing(
    monkeypatch,
):
    monkeypatch.setattr(config, "SEARCH_MODE", "sparse")
    monkeypatch.setattr(config, "RERANK_ENABLED", False)

    calls = []

    class FakeRouter:
        def is_complex(self, query):
            calls.append(("route", query))
            return True

    class FakeRewriter:
        def rewrite(self, query):
            calls.append(("rewrite", query))
            raise AssertionError(
                "Rewriter must not run for a complex query"
            )

    class FakeDecomposer:
        def decompose(self, query):
            calls.append(("decompose", query))
            return [
                "sub-question one",
                "sub-question two",
            ]

    class FakeExpander:
        def expand(self, query):
            calls.append(("expand", query))
            return [
                f"{query} variant",
            ]

    processor = DefaultQueryProcessor(
        complexity_router=FakeRouter(),
        rewriter=FakeRewriter(),
        decomposer=FakeDecomposer(),
        expander=FakeExpander(),
    )

    monkeypatch.setattr(
        retrieval,
        "get_query_processor",
        lambda: processor,
    )

    retrieval_queries = []

    def fake_keyword_search(query, top_k):
        retrieval_queries.append(query)

        return [
            ("chunk-1", -1.0),
        ]

    monkeypatch.setattr(
        retrieval,
        "keyword_search",
        fake_keyword_search,
    )

    monkeypatch.setattr(
        retrieval,
        "get_chunks_with_parent_by_ids",
        lambda ids: {
            "chunk-1": {
                "parent_chunk_id": "parent-1",
                "content": "Relevant content",
                "source": "doc.txt",
            }
        },
    )

    monkeypatch.setattr(
        retrieval,
        "multi_query_fusion",
        lambda query_results, k, top_k: [
            ("chunk-1", 0.5),
        ],
    )

    results = retrieval.retrieve(
        "complex original question",
        top_k=1,
    )

    assert calls == [
        ("route", "complex original question"),
        ("decompose", "complex original question"),
        ("expand", "sub-question one"),
        ("expand", "sub-question two"),
    ]

    assert retrieval_queries == [
        "sub-question one variant",
        "sub-question two variant",
    ]

    assert len(results) == 1
    assert results[0]["score_type"] == "multi_query"