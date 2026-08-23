from rag.storage import db


def test_document_and_chunk_roundtrip():
    db.insert_document("doc-1", "file.txt", "file.txt", "hash123", version=1)
    db.insert_parent_chunk("parent-1", "doc-1", 0, "parent text here", "phash")
    db.insert_small_chunk("small-1", "doc-1", "parent-1", 0, "small text", "small text", "shash")

    fetched = db.get_chunks_with_parent_by_ids(["small-1"])
    assert fetched["small-1"]["content"] == "parent text here"   # parent expansion happened
    assert fetched["small-1"]["source"] == "file.txt"


def test_exact_content_hash_dedup_lookup():
    db.insert_document("doc-1", "file.txt", "file.txt", "hash123")
    db.insert_parent_chunk("parent-1", "doc-1", 0, "parent", "phash")
    db.insert_small_chunk("small-1", "doc-1", "parent-1", 0, "same text", "same text", "abc123")

    assert db.get_chunk_by_content_hash("abc123") == "small-1"
    assert db.get_chunk_by_content_hash("nonexistent") is None


def test_global_document_hash_lookup_is_filename_independent():
    db.insert_document("doc-1", "original_name.txt", "original_name.txt", "shared-hash-123")
    # Different source_uri/filename, same doc_hash -- should still be found.
    assert db.get_document_by_content_hash("shared-hash-123") == "doc-1"
    assert db.get_document_by_content_hash("no-such-hash") is None


def test_versioning_soft_delete():
    db.insert_document("doc-1", "file.txt", "file.txt", "hash-v1", version=1)
    db.insert_small_chunk("s1", "doc-1", None, 0, "v1 text", "v1 text", "h1")

    indexed_chunk_ids = db.mark_document_stale("doc-1")
    assert indexed_chunk_ids == ["s1"]

    latest = db.get_latest_document_by_source("file.txt")
    assert latest is None   # the only version that existed is now stale