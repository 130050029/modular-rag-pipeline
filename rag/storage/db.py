"""
db.py -- SQLite connection + schema. The "document store" half of the
two-store design.

NOTE: schema changed again -- delete rag.db before running.
"""

import sqlite3
from config import DB_PATH


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_id             TEXT PRIMARY KEY,
        filename           TEXT,
        source_uri         TEXT,          -- identity key across versions (filename, for this toy project)
        doc_hash           TEXT,          -- exact-duplicate detection (no longer globally UNIQUE --
                                           -- same hash can recur across different source_uri, or after
                                           -- a doc is revived; uniqueness is checked per source_uri instead)
        near_dup_of_doc_id TEXT NULL,
        version            INTEGER DEFAULT 1,
        is_stale           INTEGER DEFAULT 0,   -- 1 once a newer version has replaced this one
        ingested_at        TEXT
    );

    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id              TEXT PRIMARY KEY,
        doc_id                TEXT,
        parent_chunk_id       TEXT NULL,
        chunk_type            TEXT,        -- 'parent' or 'small'
        position               INTEGER,
        content                TEXT,        -- what gets shown to the LLM
        embedding_text          TEXT,        -- what actually gets embedded (usually == content;
                                             -- differs for tables, see tables.py)
        content_hash           TEXT,
        duplicate_of_chunk_id  TEXT NULL,
        is_stale               INTEGER DEFAULT 0,
        FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
    );

    CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash);
    CREATE INDEX IF NOT EXISTS idx_documents_source_uri ON documents(source_uri);
    CREATE INDEX IF NOT EXISTS idx_documents_doc_hash ON documents(doc_hash);
    """)
    conn.commit()
    conn.close()


def get_document_by_content_hash(doc_hash: str):
    """GLOBAL exact-duplicate check -- byte-identical content, regardless of
    filename/source_uri. Cheap (indexed hash lookup, same cost at a million
    rows as at a thousand) and must run BEFORE near-dup/versioning logic, so
    truly identical content skips entirely instead of paying for a MinHash
    computation and a documents-table insert it doesn't need."""
    conn = get_db()
    row = conn.execute(
        "SELECT doc_id FROM documents WHERE doc_hash = ? AND is_stale = 0 LIMIT 1",
        (doc_hash,),
    ).fetchone()
    conn.close()
    return row["doc_id"] if row else None


def get_all_indexable_chunks():
    """Non-stale, non-duplicate small chunks -- what belongs in the vector index."""
    conn = get_db()
    rows = conn.execute(
        """SELECT chunk_id, embedding_text FROM chunks
           WHERE chunk_type = 'small' AND duplicate_of_chunk_id IS NULL AND is_stale = 0
           ORDER BY rowid"""
    ).fetchall()
    conn.close()
    return rows


def get_latest_document_by_source(source_uri: str):
    """Latest non-stale version of a document with this source_uri, if any."""
    conn = get_db()
    row = conn.execute(
        """SELECT doc_id, doc_hash, version FROM documents
           WHERE source_uri = ? AND is_stale = 0
           ORDER BY version DESC LIMIT 1""",
        (source_uri,),
    ).fetchone()
    conn.close()
    return row


def insert_document(doc_id: str, filename: str, source_uri: str, doc_hash: str,
                     version: int = 1, near_dup_of_doc_id: str = None):
    conn = get_db()
    conn.execute(
        """INSERT INTO documents (doc_id, filename, source_uri, doc_hash, near_dup_of_doc_id, version, ingested_at)
           VALUES (?, ?, ?, ?, ?, ?, datetime('now'))""",
        (doc_id, filename, source_uri, doc_hash, near_dup_of_doc_id, version),
    )
    conn.commit()
    conn.close()


def mark_document_stale(doc_id: str) -> list[str]:
    """Soft-deletes a document and all its chunks. Returns the chunk_ids that
    were actually in the vector index (small, non-duplicate) so the caller
    can remove their vectors too."""
    conn = get_db()
    indexed_chunk_ids = [
        r["chunk_id"] for r in conn.execute(
            """SELECT chunk_id FROM chunks
               WHERE doc_id = ? AND chunk_type = 'small' AND duplicate_of_chunk_id IS NULL AND is_stale = 0""",
            (doc_id,),
        ).fetchall()
    ]
    conn.execute("UPDATE documents SET is_stale = 1 WHERE doc_id = ?", (doc_id,))
    conn.execute("UPDATE chunks SET is_stale = 1 WHERE doc_id = ?", (doc_id,))
    conn.commit()
    conn.close()
    return indexed_chunk_ids


def insert_parent_chunk(chunk_id: str, doc_id: str, position: int, content: str, content_hash: str):
    conn = get_db()
    conn.execute(
        """INSERT INTO chunks (chunk_id, doc_id, parent_chunk_id, chunk_type, position, content, embedding_text, content_hash)
           VALUES (?, ?, NULL, 'parent', ?, ?, ?, ?)""",
        (chunk_id, doc_id, position, content, content, content_hash),
    )
    conn.commit()
    conn.close()


def insert_small_chunk(chunk_id: str, doc_id: str, parent_chunk_id: str, position: int,
                        content: str, embedding_text: str, content_hash: str,
                        duplicate_of_chunk_id: str = None):
    conn = get_db()
    conn.execute(
        """INSERT INTO chunks (chunk_id, doc_id, parent_chunk_id, chunk_type, position, content, embedding_text, content_hash, duplicate_of_chunk_id)
           VALUES (?, ?, ?, 'small', ?, ?, ?, ?, ?)""",
        (chunk_id, doc_id, parent_chunk_id, position, content, embedding_text, content_hash, duplicate_of_chunk_id),
    )
    conn.commit()
    conn.close()


def get_chunk_by_content_hash(content_hash: str):
    conn = get_db()
    row = conn.execute(
        "SELECT chunk_id FROM chunks WHERE content_hash = ? AND chunk_type = 'small' AND is_stale = 0 LIMIT 1",
        (content_hash,),
    ).fetchone()
    conn.close()
    return row["chunk_id"] if row else None


def get_chunks_with_parent_by_ids(chunk_ids: list[str]):
    """For SMALL chunk_ids (from vector search), fetch each one's PARENT content."""
    if not chunk_ids:
        return {}
    conn = get_db()
    placeholders = ",".join("?" for _ in chunk_ids)
    rows = conn.execute(
        f"""SELECT c.chunk_id, c.content AS small_content, p.content AS parent_content, d.filename
            FROM chunks c
            JOIN documents d ON c.doc_id = d.doc_id
            LEFT JOIN chunks p ON c.parent_chunk_id = p.chunk_id
            WHERE c.chunk_id IN ({placeholders})""",
        chunk_ids,
    ).fetchall()
    conn.close()
    return {
        row["chunk_id"]: {"content": row["parent_content"] or row["small_content"], "source": row["filename"]}
        for row in rows
    }