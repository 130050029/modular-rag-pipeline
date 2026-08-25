from rag.storage import db


def test_document_and_chunk_roundtrip():
    db.insert_document(
        "doc-1",
        "file.txt",
        "file.txt",
        "hash123",
        version=1,
    )

    db.insert_parent_chunk(
        "parent-1",
        "doc-1",
        0,
        "parent text here",
        "phash",
    )

    db.insert_small_chunk(
        "small-1",
        "doc-1",
        "parent-1",
        0,
        "small text",
        "small text",
        "shash",
    )

    fetched = db.get_chunks_with_parent_by_ids(
        ["small-1"]
    )

    assert fetched["small-1"]["content"] == "parent text here"
    assert fetched["small-1"]["source"] == "file.txt"
    assert fetched["small-1"]["source_uri"] == "file.txt"
    assert fetched["small-1"]["doc_id"] == "doc-1"
    assert fetched["small-1"]["parent_chunk_id"] == "parent-1"


def test_get_chunks_with_parent_by_ids_empty_input():
    assert db.get_chunks_with_parent_by_ids([]) == {}


def test_exact_content_hash_dedup_lookup():
    db.insert_document(
        "doc-1",
        "file.txt",
        "file.txt",
        "hash123",
    )

    db.insert_parent_chunk(
        "parent-1",
        "doc-1",
        0,
        "parent",
        "phash",
    )

    db.insert_small_chunk(
        "small-1",
        "doc-1",
        "parent-1",
        0,
        "same text",
        "same text",
        "abc123",
    )

    assert db.get_chunk_by_content_hash("abc123") == "small-1"
    assert db.get_chunk_by_content_hash("nonexistent") is None


def test_global_document_hash_lookup_is_filename_independent():
    db.insert_document(
        "doc-1",
        "original_name.txt",
        "original_name.txt",
        "shared-hash-123",
    )

    assert (
        db.get_document_by_content_hash("shared-hash-123")
        == "doc-1"
    )

    assert (
        db.get_document_by_content_hash("no-such-hash")
        is None
    )


def test_stale_document_is_not_returned_by_global_hash_lookup():
    db.insert_document(
        "doc-1",
        "file.txt",
        "file.txt",
        "hash123",
    )

    db.mark_document_stale("doc-1")

    assert db.get_document_by_content_hash("hash123") is None


def test_versioning_soft_delete():
    db.insert_document(
        "doc-1",
        "file.txt",
        "file.txt",
        "hash-v1",
        version=1,
    )

    db.insert_small_chunk(
        "s1",
        "doc-1",
        None,
        0,
        "v1 text",
        "v1 text",
        "h1",
    )

    indexed_chunk_ids = db.mark_document_stale(
        "doc-1"
    )

    assert indexed_chunk_ids == ["s1"]

    latest = db.get_latest_document_by_source(
        "file.txt"
    )

    assert latest is None


def test_mark_document_stale_excludes_semantic_duplicate_from_indexed_ids():
    db.insert_document(
        "doc-1",
        "file.txt",
        "file.txt",
        "hash-v1",
    )

    db.insert_small_chunk(
        "normal",
        "doc-1",
        None,
        0,
        "normal text",
        "normal text",
        "normal-hash",
    )

    db.insert_small_chunk(
        "semantic",
        "doc-1",
        None,
        1,
        "semantic duplicate text",
        "semantic duplicate text",
        "semantic-hash",
        duplicate_of_chunk_id="normal",
        duplicate_reason="semantic",
    )

    indexed_chunk_ids = db.mark_document_stale(
        "doc-1"
    )

    assert indexed_chunk_ids == ["normal"]


def test_get_all_indexable_chunks_excludes_stale_and_duplicates():
    db.insert_document(
        "doc-1",
        "a.txt",
        "a.txt",
        "hash-a",
    )

    db.insert_document(
        "doc-2",
        "b.txt",
        "b.txt",
        "hash-b",
    )

    db.insert_small_chunk(
        "normal",
        "doc-1",
        None,
        0,
        "normal",
        "normal",
        "hash-normal",
    )

    db.insert_small_chunk(
        "semantic",
        "doc-1",
        None,
        1,
        "semantic",
        "semantic",
        "hash-semantic",
        duplicate_of_chunk_id="normal",
        duplicate_reason="semantic",
    )

    db.insert_small_chunk(
        "exact",
        "doc-2",
        None,
        0,
        "exact",
        "exact",
        "hash-exact",
        duplicate_of_chunk_id="normal",
        duplicate_reason="exact",
    )

    db.insert_small_chunk(
        "stale",
        "doc-2",
        None,
        1,
        "stale",
        "stale",
        "hash-stale",
    )

    db.mark_document_stale("doc-2")

    rows = db.get_all_indexable_chunks()

    ids = [row["chunk_id"] for row in rows]

    assert "normal" in ids
    assert "semantic" not in ids
    assert "exact" not in ids
    assert "stale" not in ids


def test_get_all_document_texts_for_near_dedup_reconstructs_parent_order():
    db.insert_document(
        "doc-1",
        "file.txt",
        "file.txt",
        "hash123",
    )

    db.insert_parent_chunk(
        "parent-1",
        "doc-1",
        0,
        "first part",
        "hash-parent-1",
    )

    db.insert_parent_chunk(
        "parent-2",
        "doc-1",
        1,
        "second part",
        "hash-parent-2",
    )

    texts = db.get_all_document_texts_for_near_dedup()

    assert texts["doc-1"] == "first part second part"


def test_get_all_document_texts_for_near_dedup_excludes_stale_documents():
    db.insert_document(
        "doc-1",
        "file.txt",
        "file.txt",
        "hash123",
    )

    db.insert_parent_chunk(
        "parent-1",
        "doc-1",
        0,
        "stale content",
        "hash-parent",
    )

    db.mark_document_stale("doc-1")

    texts = db.get_all_document_texts_for_near_dedup()

    assert "doc-1" not in texts


def test_insert_small_chunk_rejects_invalid_duplicate_reason():
    import pytest

    db.insert_document(
        "doc-1",
        "file.txt",
        "file.txt",
        "hash123",
    )

    with pytest.raises(ValueError):
        db.insert_small_chunk(
            "small-1",
            "doc-1",
            None,
            0,
            "text",
            "text",
            "hash",
            duplicate_of_chunk_id="other",
            duplicate_reason="invalid",
        )


def test_insert_small_chunk_requires_duplicate_target():
    import pytest

    db.insert_document(
        "doc-1",
        "file.txt",
        "file.txt",
        "hash123",
    )

    with pytest.raises(ValueError):
        db.insert_small_chunk(
            "small-1",
            "doc-1",
            None,
            0,
            "text",
            "text",
            "hash",
            duplicate_reason="semantic",
        )


def test_get_latest_document_by_source_returns_latest_non_stale_version():
    db.insert_document(
        "doc-v1",
        "file.txt",
        "file.txt",
        "hash-v1",
        version=1,
    )

    db.insert_document(
        "doc-v2",
        "file.txt",
        "file.txt",
        "hash-v2",
        version=2,
    )

    latest = db.get_latest_document_by_source(
        "file.txt"
    )

    assert latest["doc_id"] == "doc-v2"
    assert latest["version"] == 2