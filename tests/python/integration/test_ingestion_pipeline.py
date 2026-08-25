from rag.ingestion.pipeline import ingest_document, vector_index
from rag.storage.db import get_db, keyword_search


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
    content = "Paris is the capital of France and a major European city."

    first = ingest_document("doc.txt", content)
    second = ingest_document("doc.txt", content)

    assert first["status"] == "ingested"
    assert second["status"] == "skipped_exact_duplicate"


def test_exact_duplicate_is_global_across_filenames(fake_embeddings):
    content = "Paris is the capital of France and a major European city."

    first = ingest_document("doc_a.txt", content)
    second = ingest_document("doc_b.txt", content)

    assert first["status"] == "ingested"
    assert second["status"] == "skipped_exact_duplicate"


def test_new_version_marks_previous_version_stale(fake_embeddings):
    first = ingest_document(
        "doc.txt",
        "Version one content about cats and dogs and other animals here today.",
    )
    second = ingest_document(
        "doc.txt",
        "Version two content about cats and dogs and other animals here tomorrow.",
    )

    assert first["status"] == "ingested"
    assert second["status"] == "ingested"
    assert second["version"] == 2

    conn = get_db()
    rows = conn.execute(
        "SELECT doc_id, is_stale FROM documents WHERE source_uri = ? ORDER BY version",
        ("doc.txt",),
    ).fetchall()
    conn.close()

    assert len(rows) == 2
    assert rows[0]["doc_id"] == first["doc_id"]
    assert rows[0]["is_stale"] == 1
    assert rows[1]["doc_id"] == second["doc_id"]
    assert rows[1]["is_stale"] == 0


def test_near_duplicate_document_is_skipped(fake_embeddings):
    edited = PARAGRAPH.replace("fifteen percent", "twenty percent")

    first = ingest_document("a.txt", PARAGRAPH)
    second = ingest_document("b.txt", edited)

    assert first["status"] == "ingested"
    assert second["status"] == "skipped_near_duplicate"

def test_semantic_duplicate_is_stored_for_sparse_but_not_vector(
    fake_embeddings,
    monkeypatch,
):
    from rag.ingestion import pipeline

    first_text = (
        "The 2024 remote work policy permits employees to work remotely "
        "for two days per week."
    )
    second_text = (
        "The 2025 remote work policy permits employees to work remotely "
        "for four days per week."
    )

    # Force the first document through the normal vector-indexing path.
    monkeypatch.setattr(
        "rag.ingestion.pipeline.find_semantic_duplicate",
        lambda _vector: None,
    )

    first = ingest_document(
        "policy_2024.txt",
        first_text,
    )

    assert first["status"] == "ingested"

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

    # The ingestion pipeline's live vector index must contain the first
    # chunk at this point.
    assert pipeline.vector_index.size == 1

    # Force ONLY the second document through the semantic-duplicate path.
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
    assert second["duplicate_chunks_skipped"] == 1

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
    assert second_chunk["duplicate_reason"] == "semantic"

    # Semantic duplicates remain available to sparse/BM25 retrieval.
    sparse_ids = [
        chunk_id
        for chunk_id, _ in keyword_search(
            "four days per week",
            top_k=10,
        )
    ]

    assert second_chunk["chunk_id"] in sparse_ids

    # Query the SAME live VectorIndex instance used by ingest_document().
    #
    # Do not use a module-level `vector_index` imported from
    # rag.storage.indexing because conftest.py replaces the singleton
    # during the test fixture.
    first_vector = fake_embeddings([first_text])

    vector_ids = [
        chunk_id
        for chunk_id, _ in pipeline.vector_index.search(
            first_vector,
            top_k=10,
        )
    ]

    # Original chunk is vector indexed.
    assert first_chunk["chunk_id"] in vector_ids

    # Semantic duplicate is stored + sparse searchable, but NOT vector
    # indexed.
    assert second_chunk["chunk_id"] not in vector_ids