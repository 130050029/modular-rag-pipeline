import sqlite3

import config
from rag.storage.connection import (
    Connection,
    _to_postgres_placeholders,
)


def test_sqlite_connection_uses_configured_db_path(temp_db):
    config.DB_BACKEND = "sqlite"

    conn = Connection()

    assert conn._backend == "sqlite"
    assert isinstance(conn._conn, sqlite3.Connection)

    conn.close()


def test_sqlite_execute_fetchone_and_fetchall(temp_db):
    config.DB_BACKEND = "sqlite"

    conn = Connection()

    conn.execute(
        """
        CREATE TABLE test_items (
            id INTEGER PRIMARY KEY,
            name TEXT
        )
        """
    )

    conn.execute(
        "INSERT INTO test_items (name) VALUES (?)",
        ("alpha",),
    )
    conn.execute(
        "INSERT INTO test_items (name) VALUES (?)",
        ("beta",),
    )

    conn.commit()

    row = conn.execute(
        "SELECT name FROM test_items WHERE id = ?",
        (1,),
    ).fetchone()

    assert row["name"] == "alpha"

    rows = conn.execute(
        "SELECT name FROM test_items ORDER BY id"
    ).fetchall()

    assert [row["name"] for row in rows] == [
        "alpha",
        "beta",
    ]

    conn.close()


def test_sqlite_executescript_runs_multiple_statements(temp_db):
    config.DB_BACKEND = "sqlite"

    conn = Connection()

    conn.executescript(
        """
        CREATE TABLE first_table (
            id INTEGER PRIMARY KEY
        );

        CREATE TABLE second_table (
            id INTEGER PRIMARY KEY
        );
        """
    )

    tables = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name IN ('first_table', 'second_table')
        ORDER BY name
        """
    ).fetchall()

    assert [row["name"] for row in tables] == [
        "first_table",
        "second_table",
    ]

    conn.close()


def test_commit_persists_changes(temp_db):
    config.DB_BACKEND = "sqlite"

    conn = Connection()

    conn.execute(
        "CREATE TABLE test_items (value TEXT)"
    )

    conn.execute(
        "INSERT INTO test_items (value) VALUES (?)",
        ("persisted",),
    )

    conn.commit()
    conn.close()

    conn2 = Connection()

    row = conn2.execute(
        "SELECT value FROM test_items"
    ).fetchone()

    assert row["value"] == "persisted"

    conn2.close()


def test_placeholder_conversion():
    query = """
        SELECT *
        FROM chunks
        WHERE doc_id = ?
          AND position = ?
    """

    assert _to_postgres_placeholders(query) == """
        SELECT *
        FROM chunks
        WHERE doc_id = %s
          AND position = %s
    """


def test_get_connection_uses_runtime_backend_config(temp_db, monkeypatch):
    config.DB_BACKEND = "sqlite"

    monkeypatch.setattr(
        config,
        "DB_BACKEND",
        "sqlite",
    )

    conn = Connection()

    assert conn._backend == "sqlite"

    conn.close()