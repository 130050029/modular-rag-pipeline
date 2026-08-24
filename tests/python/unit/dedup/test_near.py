from rag.dedup.near import check_near_duplicate, register_document

# NOTE: near-dup detection uses 9-word shingles (config.MINHASH_SHINGLE_SIZE) --
# meaningful for realistic document-length text, but a short one-line sentence
# doesn't have enough shingles for a single word change to stay above the
# similarity threshold (verified empirically: a 14-word sentence with one
# word changed scores ~0.77 Jaccard vs. our 0.8 threshold -- too close, and
# not representative of what this algorithm is actually designed for). These
# tests use paragraph-length text instead, matching real usage.

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


def test_near_duplicate_detected_for_similar_text():
    edited = PARAGRAPH.replace("fifteen percent", "twenty percent")
    register_document("doc-1", PARAGRAPH)
    assert check_near_duplicate(edited) == "doc-1"


def test_unrelated_text_not_flagged_as_near_duplicate():
    register_document("doc-1", PARAGRAPH)
    unrelated = (
        "Quantum computing relies on superposition and entanglement of qubits "
        "to perform certain calculations exponentially faster than classical "
        "computers for specific problem classes such as integer factorization."
    )
    assert check_near_duplicate(unrelated) is None