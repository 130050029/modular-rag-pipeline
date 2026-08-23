"""Deterministic first-pass matching between golden evidence and retrieved text.

This is intentionally a simple Phase-A evaluator. It is not a semantic
relevance model.

For normal prose, matching is based on content-token overlap.

For evidence containing numbers, exact numeric anchors are required. This
allows natural-language evidence such as:

    "Revenue was 160000 in Q4."

to match a table row such as:

    "West | 143000 | 150000 | 148000 | 160000"

without requiring the words "revenue" or "Q4" to literally occur in the row.
"""

import re
import unicodedata


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}

_NUMBER_PATTERN = re.compile(
    r"(?<!\w)"
    r"(?:\$?\d+(?:[.,]\d+)*%?)"
    r"(?!\w)"
)


def normalize_text(text: str) -> str:
    """Normalize text for lexical comparison."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s.%$-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    """Return normalized non-stopword tokens."""
    return [
        token
        for token in normalize_text(text).split()
        if token not in _STOPWORDS
    ]


def _numeric_anchors(text: str) -> set[str]:
    """Extract exact numeric values from text.

    Numeric punctuation such as commas, decimal points, currency symbols,
    and percentages is preserved for this comparison.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    return set(_NUMBER_PATTERN.findall(text))


def evidence_is_covered(
    evidence_text: str,
    chunk_text: str,
    *,
    min_token_overlap: float = 0.60,
    min_content_tokens: int = 3,
) -> bool:
    """Return whether ``chunk_text`` covers ``evidence_text``.

    Rules:

    1. Empty evidence/chunk -> False.
    2. If evidence contains numeric anchors, every numeric anchor must occur
       exactly in the retrieved chunk.
    3. If the evidence consists only of numeric information, numeric matching
       is sufficient.
    4. Otherwise use content-token overlap.
    """
    if not evidence_text or not chunk_text:
        return False

    evidence_numbers = _numeric_anchors(evidence_text)

    if evidence_numbers:
        chunk_numbers = _numeric_anchors(chunk_text)

        if not evidence_numbers.issubset(chunk_numbers):
            return False

        # Numeric evidence is sufficiently anchored by exact numeric values
        # for this first deterministic baseline.
        return True

    evidence_tokens = _tokens(evidence_text)
    chunk_tokens = set(_tokens(chunk_text))

    if not evidence_tokens or not chunk_tokens:
        return False

    if len(evidence_tokens) < min_content_tokens:
        return all(token in chunk_tokens for token in evidence_tokens)

    overlap = sum(
        token in chunk_tokens
        for token in evidence_tokens
    )

    return (
        overlap / len(evidence_tokens)
    ) >= min_token_overlap