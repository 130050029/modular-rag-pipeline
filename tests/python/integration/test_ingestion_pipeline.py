"""
Integration tests: exercise ingest_document() end to end, the way real
traffic would, rather than testing each dedup layer in isolation.

These tests are the ones most likely to catch wiring mistakes between
deduplication, storage, sparse retrieval, and vector indexing.
"""

from rag.ingestion.pipeline import ingest_document
from rag.storage.db import (
    get_db,
    keyword_search,
)


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


def test_exact_duplicate_is_skipped(fake_embeddings):
    r1 = ingest_document(
        "doc.txt",
        "Paris is the capital of France and a major European city.",
    )

    r2 = ingest_document(
        "doc.txt",
        "Paris is the capital of France and a major European city.",
    )

    assert r1["status"] == "ingested"
    assert r2["status"] == "skipped_exact_duplicate"


def test_exact_duplicate_caught_even_under_a_different_filename(
    fake_embeddings,
):
    """Same content under a different filename is still a global exact
    duplicate.
    """
    content = (
        "Paris is the capital of France and a major European city."
    )

    r1 = ingest_document(
        "doc_a.txt",
        content,
    )

    r2 = ingest_document(
        "doc_b.txt",
        content,
    )

    assert r1["status"] == "ingested"
    assert r2["status"] == "skipped_exact_duplicate"


def test_new_version_marks_old_stale_and_bumps_version(
    fake_embeddings,
):
    r1 = ingest_document(
        "doc.txt",
        "Version one content about cats and dogs and other animals here today.",
    )

    r2 = ingest_document(
        "doc.txt",
        "Version two content about cats and dogs and other animals here tomorrow.",
    )

    assert r1["status"] == "ingested"
    assert r2["status"] == "ingested"
    assert r2["version"] == 2


def test_near_duplicate_document_is_flagged(
    fake_embeddings,
):
    edited = PARAGRAPH.replace(
        "fifteen percent",
        "twenty percent",
    )

    r1 = ingest_document(
        "a.txt",
        PARAGRAPH,
    )

    r2 = ingest_document(
        "b.txt",
        edited,
    )

    assert r1["status"] == "ingested"
    assert r2["status"] == "skipped_near_duplicate"


def test_semantic_duplicate_is_stored_and_sparse_searchable_but_not_vector_indexed(
    fake_embeddings,
    monkeypatch,
):
    """
    Regression test for the semantic-deduplication architecture.

    A semantic duplicate must:

        1. remain in the chunks table;
        2. be present in sparse/BM25 retrieval;
        3. NOT be added to the vector index.

    This is important for versioned documents where dense similarity can
    consider two versions redundant while BM25 can still distinguish exact
    version-specific terms.
    """
    first_text = (
        "The 2024 remote work policy permits employees to work remotely "
        "for two days per week."
    )

    second_text = (
        "The 2025 remote work policy permits employees to work remotely "
        "for four days per week."
    )

    first = ingest_document(
        "policy_2024.txt",
        first_text,
    )

    assert first["status"] == "ingested"

    # Force the second chunk through the semantic-duplicate branch.
    #
    # We intentionally don't depend on fake embedding semantics here. The
    # purpose of this test is the ingestion wiring after semantic duplicate
    # detection has already made its decision.
    conn = get_db()

    first_chunk = conn.execute(
        """
        SELECT chunk_id
        FROM chunks
        WHERE doc_id = ?
          AND chunk_type = 'small'
        """,
        (first["doc_id"],),
    ).fetchone()

    conn.close()

    assert first_chunk is not None

    monkeypatch.setattr(
        "rag.ingestion.pipeline.find_semantic_duplicate",
        lambda _vector: (
            first_chunk["chunk_id"],
            0.99,
        ),
    )

    second = ingest_document(
        "policy_2025.txt",
        second_text,
    )

    assert second["status"] == "ingested"
    assert second["chunks"] == 1
    assert second["duplicate_chunks_skipped"] == 1

    # -----------------------------------------------------------------------
    # 1. The semantic duplicate MUST exist in the document store.
    # -----------------------------------------------------------------------
    conn = get_db()

    second_chunk = conn.execute(
        """
        SELECT
            chunk_id,
            duplicate_of_chunk_id,
            duplicate_reason
        FROM chunks
        WHERE doc_id = ?
          AND chunk_type = 'small'
        """,
        (second["doc_id"],),
    ).fetchone()

    conn.close()

    assert second_chunk is not None
    assert (
        second_chunk["duplicate_of_chunk_id"]
        == first_chunk["chunk_id"]
    )
    assert (
        second_chunk["duplicate_reason"]
        == "semantic"
    )

    # -----------------------------------------------------------------------
    # 2. It MUST be available to sparse/BM25 retrieval.
    #
    # "four days" only occurs in the 2025 document, making this a useful
    # lexical discriminator.
    # -----------------------------------------------------------------------
    sparse_hits = keyword_search(
        "four days per week",
        top_k=10,
    )

    sparse_ids = [
        chunk_id
        for chunk_id, _score in sparse_hits
    ]

    assert second_chunk["chunk_id"] in sparse_ids

    # -----------------------------------------------------------------------
    # 3. It MUST NOT have been added to the vector index.
    # -----------------------------------------------------------------------
    from rag.ingestion.pipeline import vector_index

    vector_results = vector_index.search(
        fake_embeddings([second_text]),
        top_k=10,
    )

    vector_ids = [
        chunk_id
        for chunk_id, _score in vector_results
    ]

    assert second_chunk["chunk_id"] not in vector_ids

    # The original chunk remains the vector-indexed representation.
    assert first_chunk["chunk_id"] in vector_ids