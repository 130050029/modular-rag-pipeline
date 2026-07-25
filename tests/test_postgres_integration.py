"""
test_postgres_integration.py -- the ONLY test file in this suite that
actually exercises the Postgres backend for real. Every other test forces
DB_BACKEND to "sqlite" via the autouse temp_db fixture (see conftest.py) --
deliberately, for speed and isolation. That means this file is the sole
source of real evidence that the Postgres code path in
rag/storage/connection.py actually works, not just that it looks correct.

Requires a running Postgres matching config.py's defaults -- e.g.:
    docker compose up -d

If Postgres isn't reachable, these tests SKIP (not fail) with a clear
message, so a normal `pytest tests/` run is unaffected either way. To run
just these:
    pytest tests/test_postgres_integration.py -v

CLEANUP: unlike test_redis_integration.py (which uses a dedicated Redis DB
index it can safely FLUSHDB), this file writes into your REAL configured
Postgres database -- there's no equally cheap "separate logical database"
trick for Postgres here. The postgres_backend fixture below tracks every
doc_id created and deletes it (and its chunks) on teardown, so repeated
test runs don't accumulate rows forever.
"""

import uuid
import pytest


def _postgres_reachable() -> bool:
    try:
        import psycopg
        from config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
        conn = psycopg.connect(
            host=POSTGRES_HOST, port=POSTGRES_PORT, dbname=POSTGRES_DB,
            user=POSTGRES_USER, password=POSTGRES_PASSWORD, connect_timeout=2,
        )
        conn.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_reachable(),
    reason="Postgres not reachable at config.py's configured host/port -- "
           "run `docker compose up -d` to enable these tests.",
)


@pytest.fixture
def postgres_backend(monkeypatch):
    """Overrides the autouse temp_db fixture's SQLite-forcing for just this
    file, and yields a list -- tests append every doc_id they create to it,
    so teardown can clean them up afterward."""
    monkeypatch.setattr("config.DB_BACKEND", "postgres")
    from rag.storage.db import init_db, get_db
    init_db()

    created_doc_ids: list[str] = []
    yield created_doc_ids

    if created_doc_ids:
        conn = get_db()
        placeholders = ",".join("?" for _ in created_doc_ids)
        conn.execute(f"DELETE FROM chunks WHERE doc_id IN ({placeholders})", created_doc_ids)
        conn.execute(f"DELETE FROM documents WHERE doc_id IN ({placeholders})", created_doc_ids)
        conn.commit()
        conn.close()


def test_insert_and_exact_hash_lookup_against_real_postgres(postgres_backend):
    from rag.storage.db import insert_document, get_document_by_content_hash

    doc_id = str(uuid.uuid4())
    doc_hash = str(uuid.uuid4())
    insert_document(doc_id, "postgres_test.txt", "postgres_test.txt", doc_hash, version=1)
    postgres_backend.append(doc_id)   # register for cleanup

    assert get_document_by_content_hash(doc_hash) == doc_id


def test_versioning_and_soft_delete_against_real_postgres(postgres_backend):
    from rag.storage.db import (
        insert_document, insert_small_chunk, mark_document_stale, get_latest_document_by_source,
    )

    source_uri = f"versioned_{uuid.uuid4()}.txt"
    doc_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())

    insert_document(doc_id, source_uri, source_uri, str(uuid.uuid4()), version=1)
    insert_small_chunk(chunk_id, doc_id, None, 0, "some text", "some text", str(uuid.uuid4()))
    postgres_backend.append(doc_id)   # register for cleanup

    stale_ids = mark_document_stale(doc_id)
    assert chunk_id in stale_ids
    assert get_latest_document_by_source(source_uri) is None
