from rag.evaluation.evidence_matching import evidence_is_covered


def test_exact_evidence_is_covered():
    assert evidence_is_covered(
        "Revenue was 160000 in Q4.",
        "West | 143000 | 150000 | 148000 | 160000",
    )


def test_case_and_punctuation_are_ignored():
    assert evidence_is_covered(
        "The Amazon absorbs large amounts of carbon dioxide.",
        "The forest absorbs large amounts of carbon dioxide from the atmosphere.",
    )


def test_unrelated_chunk_is_not_covered():
    assert not evidence_is_covered(
        "Revenue was 160000 in Q4.",
        "The East region recorded 99000 in Q4.",
    )


def test_short_evidence_requires_all_tokens():
    assert evidence_is_covered(
        "160000",
        "West | 143000 | 150000 | 148000 | 160000",
    )

    assert not evidence_is_covered(
        "160000",
        "West | 143000 | 150000 | 148000 | 160001",
    )
