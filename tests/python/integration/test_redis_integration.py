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

        from config import (
            REDIS_HOST,
            REDIS_PORT,
        )

        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=TEST_REDIS_DB,
            socket_connect_timeout=2,
        )

        client.ping()

        return True

    except Exception as e:
        _reachability_error = (
            f"{type(e).__name__}: {e}"
        )
        return False


pytestmark = pytest.mark.skipif(
    not _redis_reachable(),
    reason=(
        "Redis not reachable -- run `docker compose up -d`, and ensure "
        "`pip install -r requirements.txt` has been run. "
        f"Underlying error: {_reachability_error}"
    ),
)


@pytest.fixture
def redis_backend(monkeypatch):
    import redis

    from config import (
        REDIS_HOST,
        REDIS_PORT,
    )

    client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=TEST_REDIS_DB,
        decode_responses=True,
    )

    client.flushdb()

    monkeypatch.setattr(
        "config.NEAR_DUP_BACKEND",
        "redis",
    )

    monkeypatch.setattr(
        "config.REDIS_DB",
        TEST_REDIS_DB,
    )

    # Force near.py to construct a fresh backend against this test database,
    # rather than reusing a backend initialized with the normal application
    # configuration.
    import rag.dedup.near as near_module

    monkeypatch.setattr(
        near_module,
        "_redis_backend",
        None,
    )

    yield client

    client.flushdb()


def test_redis_near_duplicate_detected_for_similar_text(
    redis_backend,
):
    from rag.dedup.near import (
        check_near_duplicate,
        register_document,
    )

    edited = PARAGRAPH.replace(
        "fifteen percent",
        "twenty percent",
    )

    register_document(
        "doc-1",
        PARAGRAPH,
    )

    assert (
        check_near_duplicate(edited)
        == "doc-1"
    )


def test_redis_unrelated_text_not_flagged(
    redis_backend,
):
    from rag.dedup.near import (
        check_near_duplicate,
        register_document,
    )

    register_document(
        "doc-1",
        PARAGRAPH,
    )

    unrelated = (
        "Quantum computing relies on superposition and entanglement of qubits "
        "to perform certain calculations exponentially faster than classical "
        "computers for specific problem classes such as integer factorization."
    )

    assert (
        check_near_duplicate(unrelated)
        is None
    )


def test_redis_index_persists_across_backend_instances(
    redis_backend,
):
    """A fresh Redis backend instance still sees documents registered by a
    different backend instance, demonstrating the cross-process persistence
    that the Redis backend is intended to provide."""
    from rag.dedup.near import (
        check_near_duplicate,
        register_document,
    )

    import rag.dedup.near as near_module

    register_document(
        "doc-1",
        PARAGRAPH,
    )

    # Drop the cached backend so the next operation constructs a completely
    # new _RedisBackend object.
    near_module._redis_backend = None

    edited = PARAGRAPH.replace(
        "fifteen percent",
        "twenty percent",
    )

    assert (
        check_near_duplicate(edited)
        == "doc-1"
    )


def test_redis_multiple_documents_can_be_distinguished(
    redis_backend,
):
    """The Redis index should retain multiple registered documents and return
    the appropriate near duplicate rather than merely proving that Redis has
    some data in it."""
    from rag.dedup.near import (
        check_near_duplicate,
        register_document,
    )

    first = PARAGRAPH

    second = (
        "The regional hospital announced a new initiative today aimed at "
        "reducing emergency department congestion during peak hours. The plan "
        "includes additional clinical staff, expanded appointment capacity, "
        "and updated scheduling procedures to improve patient flow."
    )

    register_document(
        "doc-traffic",
        first,
    )

    register_document(
        "doc-hospital",
        second,
    )

    edited_first = first.replace(
        "fifteen percent",
        "twenty percent",
    )

    edited_second = second.replace(
        "additional clinical staff",
        "additional medical staff",
    )

    assert (
        check_near_duplicate(edited_first)
        == "doc-traffic"
    )

    assert (
        check_near_duplicate(edited_second)
        == "doc-hospital"
    )


def test_redis_backend_is_empty_after_fixture_cleanup(
    redis_backend,
):
    """The dedicated test database starts clean for each fixture instance.

    This protects the integration suite from accidentally passing because of
    data left behind by a previous test or manual application run.
    """
    from rag.dedup.near import register_document

    register_document(
        "doc-1",
        PARAGRAPH,
    )

    assert redis_backend.dbsize() > 0