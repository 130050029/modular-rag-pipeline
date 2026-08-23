import config as config

from rag.storage.db import (
    insert_document,
    insert_parent_chunk,
    insert_small_chunk,
    keyword_search,
)


def _insert_searchable_chunk(
    doc_id,
    chunk_id,
    text,
    position=0,
):
    insert_parent_chunk(
        f"parent-{chunk_id}",
        doc_id,
        position,
        text,
        f"hash-parent-{chunk_id}",
    )

    insert_small_chunk(
        chunk_id,
        doc_id,
        f"parent-{chunk_id}",
        position,
        text,
        text,
        f"hash-{chunk_id}",
    )


def test_sqlite_bm25_is_term_search_not_phrase_search(
    temp_db,
):
    config.DB_BACKEND = "sqlite"

    insert_document(
        "doc-a",
        "a.txt",
        "a.txt",
        "doc-hash-a",
    )

    insert_document(
        "doc-b",
        "b.txt",
        "b.txt",
        "doc-hash-b",
    )

    _insert_searchable_chunk(
        "doc-a",
        "chunk-a",
        "Amazon deforestation is driven by agriculture.",
    )

    _insert_searchable_chunk(
        "doc-b",
        "chunk-b",
        "The Amazon River is extremely long.",
    )

    results = keyword_search(
        "What causes deforestation in the Amazon?",
        top_k=10,
    )

    ids = [
        chunk_id
        for chunk_id, _ in results
    ]

    assert "chunk-a" in ids
    assert "chunk-b" in ids
    assert ids.index("chunk-a") < ids.index("chunk-b")


def test_sqlite_bm25_handles_fts_punctuation_without_crashing(
    temp_db,
):
    config.DB_BACKEND = "sqlite"

    insert_document(
        "doc-a",
        "a.txt",
        "a.txt",
        "doc-hash-a",
    )

    _insert_searchable_chunk(
        "doc-a",
        "chunk-a",
        "C++ APIs, C# APIs, and foo/bar are examples.",
    )

    results = keyword_search(
        'C++ "foo/bar" (APIs)',
        top_k=10,
    )

    assert results
    assert results[0][0] == "chunk-a"


def test_sqlite_bm25_empty_or_punctuation_only_query_returns_empty(
    temp_db,
):
    config.DB_BACKEND = "sqlite"

    insert_document(
        "doc-a",
        "a.txt",
        "a.txt",
        "doc-hash-a",
    )

    _insert_searchable_chunk(
        "doc-a",
        "chunk-a",
        "Some searchable text.",
    )

    assert keyword_search(
        "!!! --- ???",
        top_k=10,
    ) == []


def test_keyword_search_rejects_negative_top_k():
    import pytest

    with pytest.raises(ValueError):
        keyword_search(
            "anything",
            top_k=-1,
        )


def test_semantic_duplicates_remain_sparse_searchable(
    temp_db,
):
    """
    Semantic duplicates are intentionally different from exact duplicates.

    They remain in the DB and FTS/BM25 because their lexical details may be
    important even though their embeddings are considered redundant.
    """
    config.DB_BACKEND = "sqlite"

    insert_document(
        "doc-a",
        "policy_2024.txt",
        "policy_2024.txt",
        "doc-hash-a",
    )

    insert_document(
        "doc-b",
        "policy_2025.txt",
        "policy_2025.txt",
        "doc-hash-b",
    )

    _insert_searchable_chunk(
        "doc-a",
        "chunk-2024",
        "Remote work policy allows two days per week.",
    )

    insert_parent_chunk(
        "parent-chunk-2025",
        "doc-b",
        0,
        "Remote work policy allows four days per week.",
        "hash-parent-2025",
    )

    insert_small_chunk(
        "chunk-2025",
        "doc-b",
        "parent-chunk-2025",
        0,
        "Remote work policy allows four days per week.",
        "Remote work policy allows four days per week.",
        "hash-2025",
        duplicate_of_chunk_id="chunk-2024",
        duplicate_reason="semantic",
    )

    results = keyword_search(
        "four days per week",
        top_k=10,
    )

    ids = [
        chunk_id
        for chunk_id, _ in results
    ]

    assert "chunk-2025" in ids


def test_exact_duplicates_remain_excluded_from_sparse_search(
    temp_db,
):
    """
    Exact duplicate chunks retain the old behavior: they are stored for
    lineage but do not create another sparse-search result.
    """
    config.DB_BACKEND = "sqlite"

    insert_document(
        "doc-a",
        "a.txt",
        "a.txt",
        "doc-hash-a",
    )

    insert_document(
        "doc-b",
        "b.txt",
        "b.txt",
        "doc-hash-b",
    )

    _insert_searchable_chunk(
        "doc-a",
        "chunk-original",
        "Amazon deforestation agriculture.",
    )

    insert_parent_chunk(
        "parent-duplicate",
        "doc-b",
        0,
        "Amazon deforestation agriculture.",
        "hash-parent-duplicate",
    )

    insert_small_chunk(
        "chunk-duplicate",
        "doc-b",
        "parent-duplicate",
        0,
        "Amazon deforestation agriculture.",
        "Amazon deforestation agriculture.",
        "hash-duplicate",
        duplicate_of_chunk_id="chunk-original",
        duplicate_reason="exact",
    )

    results = keyword_search(
        "Amazon deforestation",
        top_k=10,
    )

    ids = [
        chunk_id
        for chunk_id, _ in results
    ]

    assert "chunk-original" in ids
    assert "chunk-duplicate" not in ids