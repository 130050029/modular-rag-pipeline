import uuid

import pytest


def _postgres_reachable() -> bool:
    try:
        import psycopg

        from config import (
            POSTGRES_HOST,
            POSTGRES_PORT,
            POSTGRES_DB,
            POSTGRES_USER,
            POSTGRES_PASSWORD,
        )

        conn = psycopg.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            connect_timeout=2,
        )
        conn.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_reachable(),
    reason=(
        "Postgres not reachable at config.py's configured host/port -- "
        "run `docker compose up -d` to enable these tests."
    ),
)


@pytest.fixture
def postgres_backend(monkeypatch):
    """Run storage tests against the real PostgreSQL backend.

    The fixture initializes the schema, tracks every document created by the
    test, and removes those documents afterward.

    The tests deliberately use UUIDs for document/chunk IDs so they cannot
    collide with another test run or with application data.
    """
    monkeypatch.setattr("config.DB_BACKEND", "postgres")

    from rag.storage.db import init_db, get_db

    init_db()

    created_doc_ids: list[str] = []

    yield created_doc_ids

    if created_doc_ids:
        conn = get_db()

        placeholders = ",".join(
            "?" for _ in created_doc_ids
        )

        # Delete chunks first because documents owns the parent rows.
        conn.execute(
            f"""
            DELETE FROM chunks
            WHERE doc_id IN ({placeholders})
            """,
            created_doc_ids,
        )

        conn.execute(
            f"""
            DELETE FROM documents
            WHERE doc_id IN ({placeholders})
            """,
            created_doc_ids,
        )

        conn.commit()
        conn.close()


def test_connection_uses_postgres_backend(postgres_backend):
    """The storage connection wrapper really connects through psycopG."""
    from rag.storage.db import get_db

    conn = get_db()

    row = conn.execute(
        "SELECT 1 AS value"
    ).fetchone()

    conn.close()

    assert row["value"] == 1


def test_insert_and_exact_hash_lookup_against_real_postgres(
    postgres_backend,
):
    from rag.storage.db import (
        insert_document,
        get_document_by_content_hash,
    )

    doc_id = str(uuid.uuid4())
    doc_hash = str(uuid.uuid4())

    insert_document(
        doc_id,
        "postgres_test.txt",
        "postgres_test.txt",
        doc_hash,
        version=1,
    )

    postgres_backend.append(doc_id)

    assert get_document_by_content_hash(doc_hash) == doc_id


def test_document_and_chunk_roundtrip_against_real_postgres(
    postgres_backend,
):
    from rag.storage.db import (
        insert_document,
        insert_parent_chunk,
        insert_small_chunk,
        get_chunks_with_parent_by_ids,
    )

    doc_id = str(uuid.uuid4())
    parent_id = str(uuid.uuid4())
    small_id = str(uuid.uuid4())

    insert_document(
        doc_id,
        "roundtrip.txt",
        "roundtrip.txt",
        str(uuid.uuid4()),
        version=1,
    )

    insert_parent_chunk(
        parent_id,
        doc_id,
        0,
        "This is the complete parent text.",
        str(uuid.uuid4()),
    )

    insert_small_chunk(
        small_id,
        doc_id,
        parent_id,
        0,
        "This is the small chunk.",
        "This is the small chunk.",
        str(uuid.uuid4()),
    )

    postgres_backend.append(doc_id)

    fetched = get_chunks_with_parent_by_ids([small_id])

    assert small_id in fetched
    assert fetched[small_id]["content"] == (
        "This is the complete parent text."
    )
    assert fetched[small_id]["source"] == "roundtrip.txt"
    assert fetched[small_id]["source_uri"] == "roundtrip.txt"
    assert fetched[small_id]["doc_id"] == doc_id
    assert fetched[small_id]["parent_chunk_id"] == parent_id


def test_versioning_and_soft_delete_against_real_postgres(
    postgres_backend,
):
    from rag.storage.db import (
        insert_document,
        insert_small_chunk,
        mark_document_stale,
        get_latest_document_by_source,
    )

    source_uri = f"versioned_{uuid.uuid4()}.txt"
    doc_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())

    insert_document(
        doc_id,
        source_uri,
        source_uri,
        str(uuid.uuid4()),
        version=1,
    )

    insert_small_chunk(
        chunk_id,
        doc_id,
        None,
        0,
        "some text",
        "some text",
        str(uuid.uuid4()),
    )

    postgres_backend.append(doc_id)

    latest_before = get_latest_document_by_source(source_uri)

    assert latest_before is not None
    assert latest_before["doc_id"] == doc_id
    assert latest_before["version"] == 1

    stale_ids = mark_document_stale(doc_id)

    assert chunk_id in stale_ids

    assert get_latest_document_by_source(source_uri) is None


def test_postgres_keyword_search_uses_tsvector(
    postgres_backend,
):
    """The PostgreSQL backend must use its generated tsvector column.

    This is deliberately different from the SQLite FTS5 tests: this verifies
    the real PostgreSQL schema and ranking path rather than merely exercising
    the backend-selection branch.
    """
    from rag.storage.db import (
        insert_document,
        insert_parent_chunk,
        insert_small_chunk,
        keyword_search,
    )

    doc_a = str(uuid.uuid4())
    doc_b = str(uuid.uuid4())

    chunk_a = str(uuid.uuid4())
    chunk_b = str(uuid.uuid4())

    insert_document(
        doc_a,
        "postgres_a.txt",
        "postgres_a.txt",
        str(uuid.uuid4()),
    )

    insert_document(
        doc_b,
        "postgres_b.txt",
        "postgres_b.txt",
        str(uuid.uuid4()),
    )

    insert_parent_chunk(
        str(uuid.uuid4()),
        doc_a,
        0,
        "Amazon deforestation is driven by agricultural expansion.",
        str(uuid.uuid4()),
    )

    insert_small_chunk(
        chunk_a,
        doc_a,
        None,
        0,
        "Amazon deforestation is driven by agricultural expansion.",
        "Amazon deforestation is driven by agricultural expansion.",
        str(uuid.uuid4()),
    )

    insert_parent_chunk(
        str(uuid.uuid4()),
        doc_b,
        0,
        "The Amazon River is one of the world's major rivers.",
        str(uuid.uuid4()),
    )

    insert_small_chunk(
        chunk_b,
        doc_b,
        None,
        0,
        "The Amazon River is one of the world's major rivers.",
        "The Amazon River is one of the world's major rivers.",
        str(uuid.uuid4()),
    )

    postgres_backend.extend([doc_a, doc_b])

    results = keyword_search(
        "Amazon deforestation",
        top_k=10,
    )

    ids = [
        chunk_id
        for chunk_id, _score in results
    ]

    assert chunk_a in ids
    assert chunk_b not in ids


def test_postgres_semantic_duplicate_remains_sparse_searchable(
    postgres_backend,
):
    """PostgreSQL must preserve the same semantic-deduplication contract
    as SQLite: semantic duplicates remain searchable through sparse retrieval.
    """
    from rag.storage.db import (
        insert_document,
        insert_parent_chunk,
        insert_small_chunk,
        keyword_search,
    )

    doc_a = str(uuid.uuid4())
    doc_b = str(uuid.uuid4())

    chunk_a = str(uuid.uuid4())
    chunk_b = str(uuid.uuid4())

    insert_document(
        doc_a,
        "policy_2024.txt",
        "policy_2024.txt",
        str(uuid.uuid4()),
    )

    insert_document(
        doc_b,
        "policy_2025.txt",
        "policy_2025.txt",
        str(uuid.uuid4()),
    )

    insert_parent_chunk(
        str(uuid.uuid4()),
        doc_a,
        0,
        "Remote work policy allows two days per week.",
        str(uuid.uuid4()),
    )

    insert_small_chunk(
        chunk_a,
        doc_a,
        None,
        0,
        "Remote work policy allows two days per week.",
        "Remote work policy allows two days per week.",
        str(uuid.uuid4()),
    )

    insert_parent_chunk(
        str(uuid.uuid4()),
        doc_b,
        0,
        "Remote work policy allows four days per week.",
        str(uuid.uuid4()),
    )

    insert_small_chunk(
        chunk_b,
        doc_b,
        None,
        0,
        "Remote work policy allows four days per week.",
        "Remote work policy allows four days per week.",
        str(uuid.uuid4()),
        duplicate_of_chunk_id=chunk_a,
        duplicate_reason="semantic",
    )

    postgres_backend.extend([doc_a, doc_b])

    results = keyword_search(
        "four days per week",
        top_k=10,
    )

    ids = [
        chunk_id
        for chunk_id, _score in results
    ]

    assert chunk_b in ids


def test_postgres_exact_duplicate_remains_excluded_from_sparse_search(
    postgres_backend,
):
    """Exact duplicate chunks remain stored but are excluded from sparse
    retrieval, matching the SQLite architecture."""
    from rag.storage.db import (
        insert_document,
        insert_parent_chunk,
        insert_small_chunk,
        keyword_search,
    )

    doc_a = str(uuid.uuid4())
    doc_b = str(uuid.uuid4())

    chunk_a = str(uuid.uuid4())
    chunk_b = str(uuid.uuid4())

    insert_document(
        doc_a,
        "original.txt",
        "original.txt",
        str(uuid.uuid4()),
    )

    insert_document(
        doc_b,
        "duplicate.txt",
        "duplicate.txt",
        str(uuid.uuid4()),
    )

    insert_parent_chunk(
        str(uuid.uuid4()),
        doc_a,
        0,
        "Amazon deforestation agriculture.",
        str(uuid.uuid4()),
    )

    insert_small_chunk(
        chunk_a,
        doc_a,
        None,
        0,
        "Amazon deforestation agriculture.",
        "Amazon deforestation agriculture.",
        str(uuid.uuid4()),
    )

    insert_parent_chunk(
        str(uuid.uuid4()),
        doc_b,
        0,
        "Amazon deforestation agriculture.",
        str(uuid.uuid4()),
    )

    insert_small_chunk(
        chunk_b,
        doc_b,
        None,
        0,
        "Amazon deforestation agriculture.",
        "Amazon deforestation agriculture.",
        str(uuid.uuid4()),
        duplicate_of_chunk_id=chunk_a,
        duplicate_reason="exact",
    )

    postgres_backend.extend([doc_a, doc_b])

    results = keyword_search(
        "Amazon deforestation",
        top_k=10,
    )

    ids = [
        chunk_id
        for chunk_id, _score in results
    ]

    assert chunk_a in ids
    assert chunk_b not in ids