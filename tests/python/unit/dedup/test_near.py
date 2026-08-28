from rag.dedup.near import check_near_duplicate, register_document

# NOTE: near-dup detection uses 9-word shingles (config.MINHASH_SHINGLE_SIZE) --
# meaningful for realistic document-length text, but a short one-line sentence
# doesn't have enough shingles for a single word change to stay above the
# similarity threshold (verified empirically: a 14-word sentence with one
# word changed scores ~0.77 Jaccard vs. our 0.8 threshold -- too close, and
# not representative of what this algorithm is actually designed for). These
# tests use paragraph-length text instead, matching real usage.

import config

from rag.dedup import near


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


def test_near_duplicate_detected_for_similar_text(monkeypatch):
    monkeypatch.setattr(config, "NEAR_DUP_BACKEND", "memory")

    edited = PARAGRAPH.replace("fifteen percent", "twenty percent")

    near._memory_backend = near._MemoryBackend()

    near.register_document("doc-1", PARAGRAPH)

    assert near.check_near_duplicate(edited) == "doc-1"


def test_unrelated_text_not_flagged_as_near_duplicate(monkeypatch):
    monkeypatch.setattr(config, "NEAR_DUP_BACKEND", "memory")

    near._memory_backend = near._MemoryBackend()

    near.register_document("doc-1", PARAGRAPH)

    unrelated = (
        "Quantum computing relies on superposition and entanglement of qubits "
        "to perform certain calculations exponentially faster than classical "
        "computers for specific problem classes such as integer factorization."
    )

    assert near.check_near_duplicate(unrelated) is None


def test_exact_text_is_detected_as_near_duplicate(monkeypatch):
    monkeypatch.setattr(config, "NEAR_DUP_BACKEND", "memory")

    near._memory_backend = near._MemoryBackend()

    near.register_document("doc-1", PARAGRAPH)

    assert near.check_near_duplicate(PARAGRAPH) == "doc-1"


def test_short_text_uses_single_shingle():
    result = near._shingles("one two")

    assert result == {"one two"}


def test_short_text_minhash_is_computable():
    minhash = near.compute_minhash("one two")

    assert minhash is not None
    assert len(minhash.hashvalues) == config.MINHASH_NUM_PERM


def test_register_multiple_documents_returns_registered_match(monkeypatch):
    monkeypatch.setattr(config, "NEAR_DUP_BACKEND", "memory")

    near._memory_backend = near._MemoryBackend()

    first = (
        "The regional transit authority announced expanded bus service "
        "for downtown commuters during weekday peak hours."
    )

    second = (
        "The public library announced a new digital lending program "
        "for residents across the metropolitan area."
    )

    near.register_document("doc-transit", first)
    near.register_document("doc-library", second)

    edited_first = first.replace(
        "The regional transit",
        "regional transit",
    )

    assert near.check_near_duplicate(edited_first) == "doc-transit"


def test_memory_backend_is_used_when_configured(monkeypatch):
    monkeypatch.setattr(config, "NEAR_DUP_BACKEND", "memory")

    backend = near._MemoryBackend()
    monkeypatch.setattr(near, "_memory_backend", backend)

    near.register_document("doc-1", PARAGRAPH)

    edited = PARAGRAPH.replace(
        "fifteen percent",
        "twenty percent",
    )

    assert near.check_near_duplicate(edited) == "doc-1"