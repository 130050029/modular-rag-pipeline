"""Query complexity routing for Phase B query intelligence."""

import re
from typing import Protocol


class QueryComplexityRouter(Protocol):
    """Determine whether a query should use the complex-query path."""

    def is_complex(self, query: str) -> bool:
        """Return True when the query should be decomposed."""
        ...


class DefaultQueryComplexityRouter:
    """Deterministic query-complexity router.

    The router identifies queries that clearly contain multiple reasoning
    tasks or explicit comparison requirements.

    This is intentionally conservative. It is a baseline router, not an
    attempt to perform semantic understanding. A future classifier can
    replace this implementation without changing the QueryProcessor
    contract.
    """

    _COMPARISON_PATTERNS = (
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bcompar(e|ing|ed)\b",
        r"\bcontrast\b",
        r"\bdifference(?:s)?\b",
        r"\bdiffer(?:s|ence)?\b",
        r"\bversus\b",
        r"\bvs\.?\b",
    )

    _MULTI_TASK_PATTERNS = (
        r"\band\s+explain\b",
        r"\band\s+compare\b",
        r"\band\s+contrast\b",
        r"\band\s+describe\b",
        r"\band\s+summarize\b",
        r"\band\s+list\b",
        r"\band\s+identify\b",
        r"\band\s+calculate\b",
        r"\band\s+recommend\b",
    )

    _MULTI_QUESTION_PATTERN = r"\?.+\?"

    def is_complex(self, query: str) -> bool:
        """Return True when the query contains clear complexity signals."""

        normalized = query.strip().lower()

        if not normalized:
            return False

        if self._contains_comparison_signal(normalized):
            return True

        if self._contains_multi_task_signal(normalized):
            return True

        if re.search(self._MULTI_QUESTION_PATTERN, normalized):
            return True

        return False

    def _contains_comparison_signal(self, query: str) -> bool:
        return any(
            re.search(pattern, query)
            for pattern in self._COMPARISON_PATTERNS
        )

    def _contains_multi_task_signal(self, query: str) -> bool:
        return any(
            re.search(pattern, query)
            for pattern in self._MULTI_TASK_PATTERNS
        )