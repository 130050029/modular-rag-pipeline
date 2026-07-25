"""
connection.py -- database backend abstraction.

Supports SQLite (default, zero-setup) and Postgres (config.DB_BACKEND =
"postgres", via psycopg3), behind one uniform interface, so db.py's query
functions never need to know or care which backend is active.

Two things make this possible without pulling in a full ORM:

  1. Placeholder translation. SQLite uses "?" placeholders; Postgres (both
     psycopg2 and psycopg3) uses "%s". db.py always WRITES queries using
     "?" -- this file translates them automatically when the active
     backend is Postgres, so no query in db.py needs two versions.

  2. Dict-like row access. sqlite3.Row supports row["column_name"] access
     natively. psycopg3 achieves the same by passing row_factory=dict_row
     at connection time (a cleaner mechanism than psycopg2's separate
     RealDictCursor class) -- either way, every `row["chunk_id"]`-style
     line in db.py works unchanged regardless of backend.

What this does NOT attempt to solve (be aware of the limits): schema
migrations, connection pooling, and any SQL dialect difference beyond
placeholders and multi-statement scripts. Our schema only uses TEXT and
INTEGER columns, which both backends interpret compatibly -- a more complex
schema might need real dialect-specific handling, at which point reaching
for SQLAlchemy (or another real ORM/query layer) would be the more robust
answer than extending this by hand.
"""

import re
import sqlite3   # always available (stdlib) -- imported unconditionally so
                  # a runtime-patched DB_BACKEND (e.g. in tests) can never
                  # find this name missing, regardless of what DB_BACKEND
                  # was set to when this module was first imported.
from config import DB_PATH


_PLACEHOLDER_RE = re.compile(r"\?")


def _to_postgres_placeholders(query: str) -> str:
    return _PLACEHOLDER_RE.sub("%s", query)


class Connection:
    """Wraps either a sqlite3 or psycopg3 connection behind one interface:
    execute(), executescript(), fetchone(), fetchall(), commit(), close().
    Callers always write queries using "?" placeholders regardless of which
    backend is actually active.

    DB_BACKEND is read fresh from config on every Connection creation (not
    cached at module-import time) so tests can reliably force "sqlite" via
    monkeypatch regardless of what's set in the real environment -- and
    psycopg is only imported here, lazily, the moment a Postgres connection
    is actually requested, rather than at module load time. This is what
    fixes a real failure mode: previously, if DB_BACKEND=postgres was set in
    the shell when pytest started, this module would import psycopg only
    and never sqlite3 -- so even though tests correctly patched DB_BACKEND
    back to "sqlite" at runtime, the sqlite3 name simply didn't exist yet,
    producing a NameError. Importing sqlite3 unconditionally above, and
    psycopg lazily below, removes that failure mode entirely.
    """

    def __init__(self):
        from config import DB_BACKEND   # re-imported here, not just at module
                                          # top, so a monkeypatched config
                                          # value is picked up correctly

        self._backend = DB_BACKEND

        if self._backend == "postgres":
            import psycopg
            from psycopg.rows import dict_row
            from config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

            self._conn = psycopg.connect(
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                dbname=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                row_factory=dict_row,   # psycopg3's mechanism for row["col"] access,
                                        # replacing psycopg2's separate RealDictCursor class
            )
            self._cursor = self._conn.cursor()
        else:
            self._conn = sqlite3.connect(DB_PATH)
            self._conn.row_factory = sqlite3.Row
            self._cursor = self._conn.cursor()

    def execute(self, query: str, params: tuple = ()):
        if self._backend == "postgres":
            query = _to_postgres_placeholders(query)
        self._cursor.execute(query, params)
        return self._cursor

    def executescript(self, script: str):
        # sqlite3 has a dedicated executescript() for multi-statement DDL;
        # psycopg3's execute() already handles multiple ";"-separated
        # statements in one call (same as psycopg2 did), so a plain
        # execute() covers Postgres here -- only SQLite needs the branch.
        if self._backend == "postgres":
            self._cursor.execute(script)
        else:
            self._conn.executescript(script)

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()


def get_connection() -> Connection:
    return Connection()