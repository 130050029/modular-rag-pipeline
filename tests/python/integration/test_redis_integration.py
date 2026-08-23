"""
test_redis_integration.py -- exercises the Redis-backed near-duplicate
detection for real, the same way test_postgres_integration.py exercises
Postgres. Every other test forces config.NEAR_DUP_BACKEND to "memory" (see
conftest.py's fresh_near_dedup_index fixture) -- this file deliberately
overrides that.

Requires a running Redis matching config.py's defaults, e.g.:
    docker compose up -d

If Redis isn't reachable, these tests SKIP (not fail), so a normal
`pytest tests/` run is unaffected either way.

Uses Redis logical DB 15 (a common convention for a "test" database) rather
than the real configured REDIS_DB, and flushes it before/after each test --
this never touches real data even if pointed at a shared Redis instance,
since DB 15 is a completely separate keyspace within the same server.
"""

import pytest

TEST_REDIS_DB = 15

PARAGRAPH = (
    "The city council announced a new initiative today aimed at reducing traffic "
    "congestion during peak commuting hours across the downtown core. The plan "
    "includes expanded bus routes along the main corridor, additional bike lanes "
    "connecting the northern and southern districts, and updated traffic signal "
    "timing at every major intersection to improve overall flow through the city "
    "center during rush hour. Officials estimate the changes could reduce average "
    "commute times by nearly fifteen percent once fully implemented sometime next "
    "year, based on modeling done in partnership with the regional transit authority "
    "and several independent traffic consultants who reviewed the proposal in detail "
    "over the past several months before it was brought before the full council for "
    "a final vote. Public transit ridership in the metropolitan area has already "
    "increased steadily over the past three years according to internal reports, and "
    "city planners believe this new round of investment will accelerate that existing "
    "trend even further over the coming decade as more residents shift away from "
    "single occupancy vehicle commuting toward these newly expanded alternatives."
)


_reachability_error = None


def _redis_reachable() -> bool:
    global _reachability_error
    try:
        import redis
        from config import REDIS_HOST, REDIS_PORT
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=TEST_REDIS_DB, socket_connect_timeout=2)
        client.ping()
        return True
    except Exception as e:
        # Captured (not silently discarded) so the skip reason below
        # explains WHY -- e.g. "No module named 'redis'" (package not
        # installed) is a completely different problem than "Connection
        # refused" (container not running), and a bare pass/False here
        # makes that impossible to tell apart from the pytest output alone.
        _reachability_error = f"{type(e).__name__}: {e}"
        return False


pytestmark = pytest.mark.skipif(
    not _redis_reachable(),
    reason=f"Redis not reachable -- run `docker compose up -d`, and ensure "
           f"`pip install -r requirements.txt` has been run. "
           f"Underlying error: {_reachability_error}",
)


@pytest.fixture
def redis_backend(monkeypatch):
    import redis
    from config import REDIS_HOST, REDIS_PORT

    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=TEST_REDIS_DB, decode_responses=True)
    client.flushdb()   # clean slate, isolated to DB 15 specifically

    monkeypatch.setattr("config.NEAR_DUP_BACKEND", "redis")
    monkeypatch.setattr("config.REDIS_DB", TEST_REDIS_DB)

    # Force near.py to build a fresh _RedisBackend against THIS test db,
    # rather than reusing a cached backend instance pointed at the real
    # configured REDIS_DB from an earlier test or a previous run.
    import rag.dedup.near as near_module
    monkeypatch.setattr(near_module, "_redis_backend", None)

    yield client
    client.flushdb()


def test_redis_near_duplicate_detected_for_similar_text(redis_backend):
    from rag.dedup.near import check_near_duplicate, register_document

    edited = PARAGRAPH.replace("fifteen percent", "twenty percent")
    register_document("doc-1", PARAGRAPH)

    assert check_near_duplicate(edited) == "doc-1"


def test_redis_unrelated_text_not_flagged(redis_backend):
    from rag.dedup.near import check_near_duplicate, register_document

    register_document("doc-1", PARAGRAPH)
    unrelated = (
        "Quantum computing relies on superposition and entanglement of qubits "
        "to perform certain calculations exponentially faster than classical "
        "computers for specific problem classes such as integer factorization."
    )
    assert check_near_duplicate(unrelated) is None


def test_redis_index_persists_across_backend_instances(redis_backend):
    """The actual point of the Redis backend: a fresh backend instance
    (simulating a different app replica, or this same app after a restart)
    still sees documents registered by a different instance -- unlike the
    in-memory backend, where each Python process has its own isolated,
    unshared index."""
    from rag.dedup.near import check_near_duplicate, register_document
    import rag.dedup.near as near_module

    register_document("doc-1", PARAGRAPH)

    # Drop the cached backend object to force a brand new _RedisBackend()
    # to be constructed on the next call, connecting fresh -- simulating a
    # separate process rather than reusing the same in-memory Python object.
    near_module._redis_backend = None

    edited = PARAGRAPH.replace("fifteen percent", "twenty percent")
    assert check_near_duplicate(edited) == "doc-1"
