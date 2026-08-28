
from rag.evaluation.evidence_matching import (
    evidence_is_covered,
    normalize_text,
)


def test_normalize_text_lowercases_and_collapses_whitespace(): 
    assert normalize_text(" Revenue WAS\n$160,000! ") == "revenue was $160 000"


def test_normalize_text_applies_unicode_normalization():
    assert normalize_text("café") == normalize_text("cafe\u0301")


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


def test_stopwords_do_not_affect_content_overlap():
    assert evidence_is_covered(
        "The company increased revenue in the quarter.",
        "Company increased revenue during quarter.",
    )


def test_unrelated_chunk_is_not_covered():
    assert not evidence_is_covered(
        "Revenue was 160000 in Q4.",
        "The East region recorded 99000 in Q4.",
    )


def test_empty_evidence_is_not_covered():
    assert not evidence_is_covered(
        "",
        "Useful evidence is present here.",
    )


def test_empty_chunk_is_not_covered():
    assert not evidence_is_covered(
        "Useful evidence is present here.",
        "",
    )


def test_whitespace_only_input_is_not_covered():
    assert not evidence_is_covered(
        "   ",
        "Useful evidence is present here.",
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


def test_short_non_numeric_evidence_requires_all_tokens():
    assert evidence_is_covered(
        "revenue q4",
        "West revenue q4 results",
    )

    assert not evidence_is_covered(
        "revenue q4",
        "West revenue q3 results",
    )


def test_numeric_evidence_requires_all_numeric_values(): 
    assert evidence_is_covered( 
        "Revenue was 160000 in Q4.", 
        "Revenue was 160000 in Q4.", 
    ) 
    # Q4 is not treated as a numeric anchor; 160000 is the only # numeric value required by the current deterministic matcher. 
    assert evidence_is_covered( 
        "Revenue was 160000 in Q4.", 
        "Revenue was 160000 in Q3.", 
    ) 

def test_multiple_numeric_anchors_must_all_match(): 
    assert evidence_is_covered( 
        "Revenue was 160000 and profit was 42000.", 
        "The report shows revenue of 160000 and profit of 42000.", 
    ) 
    assert not evidence_is_covered( 
        "Revenue was 160000 and profit was 42000.", 
        "The report shows revenue of 160000 and profit of 41000.", 
    )


def test_numeric_percentage_anchor_is_exact():
    assert evidence_is_covered(
        "Growth was 15%.",
        "The company reported growth of 15% this year.",
    )

    assert not evidence_is_covered(
        "Growth was 15%.",
        "The company reported growth of 15.5% this year.",
    )


def test_numeric_decimal_anchor_is_exact():
    assert evidence_is_covered(
        "The score was 3.14.",
        "The recorded score was exactly 3.14.",
    )

    assert not evidence_is_covered(
        "The score was 3.14.",
        "The recorded score was exactly 3.1401.",
    )


def test_numeric_comma_and_currency_format_is_preserved():
    assert evidence_is_covered(
        "Revenue was $160,000.",
        "Revenue for the region was $160,000.",
    )

    assert not evidence_is_covered(
        "Revenue was $160,000.",
        "Revenue for the region was $160,001.",
    )


def test_numeric_evidence_is_anchored_by_exact_value_not_word_overlap():
    # Numeric evidence intentionally uses exact numeric anchors as the
    # deterministic baseline. Surrounding wording is not required.
    assert evidence_is_covered(
        "Revenue was 160000.",
        "The West region recorded 160000.",
    )


def test_content_overlap_at_threshold_is_covered():
    assert evidence_is_covered(
        "alpha beta gamma delta",
        "alpha beta gamma unrelated",
        min_token_overlap=0.75,
    )


def test_content_overlap_below_threshold_is_not_covered():
    assert not evidence_is_covered(
        "alpha beta gamma delta",
        "alpha beta unrelated unrelated",
        min_token_overlap=0.75,
    )


def test_custom_min_content_tokens_uses_all_tokens_for_short_evidence():
    assert evidence_is_covered(
        "alpha beta gamma",
        "alpha beta gamma",
        min_content_tokens=4,
    )

    assert not evidence_is_covered(
        "alpha beta gamma",
        "alpha beta unrelated",
        min_content_tokens=4,
    )