"""LLM-as-a-judge evaluation contract."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMJudgeResult:
    """Structured result returned by an LLM judge."""

    correctness: float
    groundedness: float
    relevance: float
    completeness: float
    reasoning: str = ""


class LLMJudge:
    """
    Contract for evaluating a generated answer with an LLM judge.

    The judge evaluates four dimensions:

    - correctness: does the answer correctly answer the question?
    - groundedness: are the claims supported by the supplied context?
    - relevance: does the answer stay focused on the question?
    - completeness: does the answer cover the supported parts of the question?

    The initial implementation is intentionally provider-agnostic.
    A concrete LLM-backed implementation will be added separately.
    """

    def evaluate(
        self,
        query: str,
        answer: str,
        context: list[dict],
        reference_answer: str | None = None,
    ) -> LLMJudgeResult:
        """
        Evaluate one generated answer.

        This method defines the contract that concrete judge implementations
        must satisfy.
        """

        raise NotImplementedError(
            "LLMJudge.evaluate() requires a concrete judge implementation."
        )