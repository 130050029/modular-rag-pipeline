"""
db.py -- schema + queries for the "document store" half of the two-store
design.

Backend-agnostic: works against SQLite or Postgres depending on
config.DB_BACKEND, via rag.storage.connection.Connection.

Duplicate semantics:

    exact duplicate chunk
        -> duplicate_of_chunk_id is set
        -> duplicate_reason = "exact"
        -> searchable by neither vector nor sparse retrieval

    semantic duplicate chunk
        -> duplicate_of_chunk_id is set
        -> duplicate_reason = "semantic"
        -> searchable by sparse retrieval
        -> excluded from vector index

This distinction is intentional.

A semantic duplicate can still contain lexically important information that
the embedding-based vector index considers redundant. Versioned documents are
the important example: two policy versions can be semantically very similar
while containing different dates, numbers, limits, or other exact terms that
BM25 should still be able to retrieve.
"""

import re
from datetime import datetime, timezone

from rag.storage.connection import get_connection, Connection


def get_db() -> Connection:
    return get_connection()


def init_db():
    conn = get_db()

    conn.executescript("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_id             TEXT PRIMARY KEY,
        filename           TEXT,
        source_uri         TEXT,
        doc_hash           TEXT,
        near_dup_of_doc_id TEXT NULL,
        version            INTEGER DEFAULT 1,
        is_stale           INTEGER DEFAULT 0,
        ingested_at        TEXT
    );

    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id              TEXT PRIMARY KEY,
        doc_id                TEXT,
        parent_chunk_id       TEXT NULL,
        chunk_type            TEXT,
        position              INTEGER,
        content               TEXT,
        embedding_text        TEXT,
        content_hash          TEXT,
        duplicate_of_chunk_id TEXT NULL,
        duplicate_reason      TEXT NULL,
        is_stale              INTEGER DEFAULT 0,
        FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
    );

    CREATE INDEX IF NOT EXISTS idx_chunks_content_hash
        ON chunks(content_hash);

    CREATE INDEX IF NOT EXISTS idx_documents_source_uri
        ON documents(source_uri);

    CREATE INDEX IF NOT EXISTS idx_documents_doc_hash
        ON documents(doc_hash);
    """)

    conn.commit()
    conn.close()

    # -----------------------------------------------------------------------
    # Lightweight schema migration for databases created before
    # duplicate_reason existed.
    # -----------------------------------------------------------------------
    from config import DB_BACKEND

    conn = get_db()

    if DB_BACKEND == "postgres":
        conn.executescript("""
        ALTER TABLE chunks
            ADD COLUMN IF NOT EXISTS duplicate_reason TEXT;
        """)
    else:
        columns = conn.execute(
            "PRAGMA table_info(chunks)"
        ).fetchall()

        column_names = {
            row["name"]
            for row in columns
        }

        if "duplicate_reason" not in column_names:
            conn.execute(
                "ALTER TABLE chunks ADD COLUMN duplicate_reason TEXT"
            )

    conn.commit()
    conn.close()

    # -----------------------------------------------------------------------
    # Keyword-search schema.
    #
    # SQLite uses a separate FTS5 table.
    # Postgres uses a generated tsvector column.
    # -----------------------------------------------------------------------
    conn = get_db()

    if DB_BACKEND == "postgres":
        conn.executescript("""
        ALTER TABLE chunks
            ADD COLUMN IF NOT EXISTS content_tsv tsvector
            GENERATED ALWAYS AS (
                to_tsvector('english', content)
            ) STORED;

        CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv
            ON chunks USING GIN(content_tsv);
        """)
    else:
        conn.executescript("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
        USING fts5(
            chunk_id UNINDEXED,
            content
        );
        """)

    conn.commit()
    conn.close()


def get_document_by_content_hash(doc_hash: str):
    """GLOBAL exact-duplicate check.

    Byte-identical content is skipped regardless of filename/source_uri.
    """
    conn = get_db()

    row = conn.execute(
        """
        SELECT doc_id
        FROM documents
        WHERE doc_hash = ?
          AND is_stale = 0
        LIMIT 1
        """,
        (doc_hash,),
    ).fetchone()

    conn.close()

    return row["doc_id"] if row else None


def get_all_indexable_chunks():
    """Return SMALL chunks that belong in the vector index.

    Semantic duplicates remain in the document store and sparse index, but
    deliberately do not belong in the vector index.

    Therefore duplicate_of_chunk_id IS NULL remains the definition of a
    vector-indexable chunk.
    """
    conn = get_db()

    rows = conn.execute(
        """
        SELECT chunk_id, embedding_text
        FROM chunks
        WHERE chunk_type = 'small'
          AND duplicate_of_chunk_id IS NULL
          AND is_stale = 0
        ORDER BY chunk_id
        """
    ).fetchall()

    conn.close()

    return rows


def get_latest_document_by_source(source_uri: str):
    """Latest non-stale version of a document with this source_uri."""
    conn = get_db()

    row = conn.execute(
        """
        SELECT doc_id, doc_hash, version
        FROM documents
        WHERE source_uri = ?
          AND is_stale = 0
        ORDER BY version DESC
        LIMIT 1
        """,
        (source_uri,),
    ).fetchone()

    conn.close()

    return row


def insert_document(
    doc_id: str,
    filename: str,
    source_uri: str,
    doc_hash: str,
    version: int = 1,
    near_dup_of_doc_id: str = None,
):
    ingested_at = datetime.now(timezone.utc).isoformat()

    conn = get_db()

    conn.execute(
        """
        INSERT INTO documents (
            doc_id,
            filename,
            source_uri,
            doc_hash,
            near_dup_of_doc_id,
            version,
            ingested_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            doc_id,
            filename,
            source_uri,
            doc_hash,
            near_dup_of_doc_id,
            version,
            ingested_at,
        ),
    )

    conn.commit()
    conn.close()


def mark_document_stale(doc_id: str) -> list[str]:
    """Soft-delete a document and all its chunks.

    Returns only chunk IDs that were actually in the vector index.

    Semantic duplicate chunks are deliberately absent from this list because
    they were never added to the vector index in the first place.
    """
    conn = get_db()

    indexed_chunk_ids = [
        row["chunk_id"]
        for row in conn.execute(
            """
            SELECT chunk_id
            FROM chunks
            WHERE doc_id = ?
              AND chunk_type = 'small'
              AND duplicate_of_chunk_id IS NULL
              AND is_stale = 0
            """,
            (doc_id,),
        ).fetchall()
    ]

    conn.execute(
        """
        UPDATE documents
        SET is_stale = 1
        WHERE doc_id = ?
        """,
        (doc_id,),
    )

    conn.execute(
        """
        UPDATE chunks
        SET is_stale = 1
        WHERE doc_id = ?
        """,
        (doc_id,),
    )

    conn.commit()
    conn.close()

    return indexed_chunk_ids


def insert_parent_chunk(
    chunk_id: str,
    doc_id: str,
    position: int,
    content: str,
    content_hash: str,
):
    conn = get_db()

    conn.execute(
        """
        INSERT INTO chunks (
            chunk_id,
            doc_id,
            parent_chunk_id,
            chunk_type,
            position,
            content,
            embedding_text,
            content_hash
        )
        VALUES (?, ?, NULL, 'parent', ?, ?, ?, ?)
        """,
        (
            chunk_id,
            doc_id,
            position,
            content,
            content,
            content_hash,
        ),
    )

    conn.commit()
    conn.close()


def insert_small_chunk(
    chunk_id: str,
    doc_id: str,
    parent_chunk_id: str,
    position: int,
    content: str,
    embedding_text: str,
    content_hash: str,
    duplicate_of_chunk_id: str = None,
    duplicate_reason: str = None,
):
    if duplicate_reason not in {
        None,
        "exact",
        "semantic",
    }:
        raise ValueError(
            "duplicate_reason must be None, 'exact', or 'semantic'"
        )

    if duplicate_reason is not None and duplicate_of_chunk_id is None:
        raise ValueError(
            "duplicate_reason requires duplicate_of_chunk_id"
        )

    conn = get_db()

    conn.execute(
        """
        INSERT INTO chunks (
            chunk_id,
            doc_id,
            parent_chunk_id,
            chunk_type,
            position,
            content,
            embedding_text,
            content_hash,
            duplicate_of_chunk_id,
            duplicate_reason
        )
        VALUES (?, ?, ?, 'small', ?, ?, ?, ?, ?, ?)
        """,
        (
            chunk_id,
            doc_id,
            parent_chunk_id,
            position,
            content,
            embedding_text,
            content_hash,
            duplicate_of_chunk_id,
            duplicate_reason,
        ),
    )

    conn.commit()
    conn.close()

    # -----------------------------------------------------------------------
    # SQLite FTS5 synchronization.
    #
    # Exact duplicates are not independently searchable.
    #
    # Semantic duplicates ARE searchable here. This is the key architectural
    # distinction introduced by this change.
    # -----------------------------------------------------------------------
    from config import DB_BACKEND

    sparse_searchable = (
        duplicate_reason != "exact"
    )

    if DB_BACKEND != "postgres" and sparse_searchable:
        conn = get_db()

        conn.execute(
            """
            INSERT INTO chunks_fts (
                chunk_id,
                content
            )
            VALUES (?, ?)
            """,
            (
                chunk_id,
                content,
            ),
        )

        conn.commit()
        conn.close()


def get_chunk_by_content_hash(content_hash: str):
    conn = get_db()

    row = conn.execute(
        """
        SELECT chunk_id
        FROM chunks
        WHERE content_hash = ?
          AND chunk_type = 'small'
          AND is_stale = 0
        LIMIT 1
        """,
        (content_hash,),
    ).fetchone()

    conn.close()

    return row["chunk_id"] if row else None


def get_all_document_texts_for_near_dedup() -> dict[str, str]:
    """Reconstruct each non-stale document from its parent chunks."""
    conn = get_db()

    rows = conn.execute(
        """
        SELECT doc_id, content
        FROM chunks
        WHERE chunk_type = 'parent'
          AND is_stale = 0
        ORDER BY doc_id, position
        """
    ).fetchall()

    conn.close()

    texts_by_doc: dict[str, list[str]] = {}

    for row in rows:
        texts_by_doc.setdefault(
            row["doc_id"],
            [],
        ).append(row["content"])

    return {
        doc_id: " ".join(parts)
        for doc_id, parts in texts_by_doc.items()
    }


def get_chunks_with_parent_by_ids(chunk_ids: list[str]):
    """For SMALL chunk IDs, fetch each one's parent content."""
    if not chunk_ids:
        return {}

    conn = get_db()

    placeholders = ",".join(
        "?" for _ in chunk_ids
    )

    rows = conn.execute(
        f"""
        SELECT
            c.chunk_id,
            c.doc_id,
            c.parent_chunk_id,
            c.content AS small_content,
            p.content AS parent_content,
            d.filename,
            d.source_uri
        FROM chunks c
        JOIN documents d
          ON c.doc_id = d.doc_id
        LEFT JOIN chunks p
          ON c.parent_chunk_id = p.chunk_id
        WHERE c.chunk_id IN ({placeholders})
        """,
        chunk_ids,
    ).fetchall()

    conn.close()

    return {
        row["chunk_id"]: {
            "content": (
                row["parent_content"]
                or row["small_content"]
            ),
            "source": row["filename"],
            "source_uri": row["source_uri"],
            "doc_id": row["doc_id"],
            "parent_chunk_id": row["parent_chunk_id"],
        }
        for row in rows
    }


def keyword_search(
    query_text: str,
    top_k: int,
) -> list[tuple[str, float]]:
    """Sparse/lexical search.

    Exact duplicate chunks are excluded.

    Semantic duplicate chunks are INCLUDED because semantic deduplication
    only excludes them from vector retrieval.

    This lets BM25 recover lexical/version-specific information that dense
    semantic deduplication would otherwise hide.
    """
    from config import DB_BACKEND

    if top_k < 0:
        raise ValueError("top_k must be >= 0")

    if top_k == 0:
        return []

    conn = get_db()

    if DB_BACKEND == "postgres":
        rows = conn.execute(
            """
            SELECT
                c.chunk_id,
                ts_rank(
                    c.content_tsv,
                    plainto_tsquery('english', ?)
                ) AS score
            FROM chunks c
            WHERE c.content_tsv @@ plainto_tsquery(
                'english',
                ?
            )
              AND c.chunk_type = 'small'
              AND c.is_stale = 0
              AND (
                  c.duplicate_of_chunk_id IS NULL
                  OR c.duplicate_reason = 'semantic'
              )
            ORDER BY score DESC, c.chunk_id
            LIMIT ?
            """,
            (
                query_text,
                query_text,
                top_k,
            ),
        ).fetchall()

    else:
        # FTS5 MATCH has its own query language. Tokenize into ordinary words
        # and quote each token so user punctuation cannot inject FTS syntax.
        tokens = re.findall(
            r"[^\W_]+",
            query_text,
            flags=re.UNICODE,
        )

        if not tokens:
            conn.close()
            return []

        match_query = " OR ".join(
            f'"{token.replace(chr(34), chr(34) * 2)}"'
            for token in tokens
        )

        rows = conn.execute(
            """
            SELECT
                chunks_fts.chunk_id,
                bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c
              ON chunks_fts.chunk_id = c.chunk_id
            WHERE chunks_fts MATCH ?
              AND c.chunk_type = 'small'
              AND c.is_stale = 0
              AND (
                  c.duplicate_of_chunk_id IS NULL
                  OR c.duplicate_reason = 'semantic'
              )
            ORDER BY score ASC, chunks_fts.chunk_id
            LIMIT ?
            """,
            (
                match_query,
                top_k,
            ),
        ).fetchall()

    conn.close()

    return [
        (
            row["chunk_id"],
            float(row["score"]),
        )
        for row in rows
    ]