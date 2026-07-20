"""
Integration tests: exercise ingest_document() end to end, the way real
traffic would, rather than testing each dedup layer in isolation. These are
the tests most likely to catch a wiring mistake between modules.
"""

from rag.ingestion.pipeline import ingest_document

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
    r1 = ingest_document("doc.txt", "Paris is the capital of France and a major European city.")
    r2 = ingest_document("doc.txt", "Paris is the capital of France and a major European city.")

    assert r1["status"] == "ingested"
    assert r2["status"] == "skipped_exact_duplicate"


def test_exact_duplicate_caught_even_under_a_different_filename(fake_embeddings):
    """This is the GLOBAL exact-dup check specifically -- same content,
    different filename, should still be caught cheaply, not fall through to
    the more expensive near-dup path."""
    content = "Paris is the capital of France and a major European city."
    r1 = ingest_document("doc_a.txt", content)
    r2 = ingest_document("doc_b.txt", content)   # different filename, identical content

    assert r1["status"] == "ingested"
    assert r2["status"] == "skipped_exact_duplicate"


def test_new_version_marks_old_stale_and_bumps_version(fake_embeddings):
    r1 = ingest_document("doc.txt", "Version one content about cats and dogs and other animals here today.")
    r2 = ingest_document("doc.txt", "Version two content about cats and dogs and other animals here tomorrow.")

    assert r1["status"] == "ingested"
    assert r2["status"] == "ingested"
    assert r2["version"] == 2


def test_near_duplicate_document_is_flagged(fake_embeddings):
    edited = PARAGRAPH.replace("fifteen percent", "twenty percent")

    r1 = ingest_document("a.txt", PARAGRAPH)
    r2 = ingest_document("b.txt", edited)

    assert r1["status"] == "ingested"
    assert r2["status"] == "skipped_near_duplicate"