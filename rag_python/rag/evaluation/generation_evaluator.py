"""Generation evaluation orchestration for golden-set evaluation."""

from dataclasses import dataclass

from rag.embeddings import embed_texts
from rag.evaluation.metrics import mean
from rag.evaluation.nvidia_llm_judge import NvidiaLLMJudge

@dataclass(frozen=True)
class GenerationEvaluationResult:
    """Evaluation result for one golden-set case."""

    case_id: str
    answerable: bool
    metrics: dict[str, float]


class GenerationEvaluator:
    """Evaluate generated answers against golden-set expectations.

    The evaluator currently contains deterministic checks for:

    - correctness
    - groundedness
    - semantic answer similarity
    - citation/source compliance
    - abstention behavior

    Semantic similarity compares the generated answer with the
    golden-set reference answer using the existing embedding model.
    """
    def __init__(self, llm_judge=None):
        self.llm_judge = llm_judge

    def evaluate(
        self,
        entry: dict,
        answer: str,
        retrieved_results: list[dict],
    ) -> GenerationEvaluationResult:
        """Evaluate one generated answer."""

        judge_metrics = {}

        if self.llm_judge is not None:
            judge_result = self.llm_judge.evaluate(
                query=entry["query"],
                answer=answer,
                context=retrieved_results,
                reference_answer=entry.get("reference_answer"),
            )

            judge_metrics = {
                "judge_correctness": judge_result.correctness,
                "judge_groundedness": judge_result.groundedness,
                "judge_relevance": judge_result.relevance,
                "judge_completeness": judge_result.completeness,
            }

        answerable = entry.get("answerable", True)

        metrics = {
            "correctness": self._evaluate_correctness(
                entry,
                answer,
            ),
            "groundedness": self._evaluate_groundedness(
                entry,
                answer,
                retrieved_results,
            ),
            "citation": self._evaluate_citation(
                entry,
                answer,
                retrieved_results,
            ),
            "abstention": self._evaluate_abstention(
                entry,
                answer,
            ),
            **judge_metrics,
        }

        if answerable:
            metrics["semantic_similarity"] = (
                self._evaluate_semantic_similarity(
                    entry,
                    answer,
                )
            )

        return GenerationEvaluationResult(
            case_id=entry["id"],
            answerable=answerable,
            metrics=metrics,
        )

    def aggregate(
        self,
        results: list[GenerationEvaluationResult],
    ) -> dict[str, float]:
        """Aggregate generation metrics across applicable cases."""

        answerable_correctness = [
            result.metrics["correctness"]
            for result in results
            if result.answerable and "correctness" in result.metrics
        ]

        answerable_groundedness = [
            result.metrics["groundedness"]
            for result in results
            if result.answerable and "groundedness" in result.metrics
        ]

        answerable_semantic_similarity = [
            result.metrics["semantic_similarity"]
            for result in results
            if result.answerable
            and "semantic_similarity" in result.metrics
        ]

        answerable_citation = [
            result.metrics["citation"]
            for result in results
            if result.answerable and "citation" in result.metrics
        ]

        unanswerable_abstention = [
            result.metrics["abstention"]
            for result in results
            if not result.answerable and "abstention" in result.metrics
        ]

        judge_correctness = [
            result.metrics["judge_correctness"]
            for result in results
            if result.answerable
            and "judge_correctness" in result.metrics
        ]

        judge_groundedness = [
            result.metrics["judge_groundedness"]
            for result in results
            if result.answerable
            and "judge_groundedness" in result.metrics
        ]

        judge_relevance = [
            result.metrics["judge_relevance"]
            for result in results
            if result.answerable
            and "judge_relevance" in result.metrics
        ]

        judge_completeness = [
            result.metrics["judge_completeness"]
            for result in results
            if result.answerable
            and "judge_completeness" in result.metrics
        ]

        return {
            "citation": mean(answerable_citation),
            "abstention": mean(unanswerable_abstention),
            "correctness": mean(answerable_correctness),
            "groundedness": mean(answerable_groundedness),
            "semantic_similarity": mean(answerable_semantic_similarity),
            "judge_correctness": mean(judge_correctness),
            "judge_groundedness": mean(judge_groundedness),
            "judge_relevance": mean(judge_relevance),
            "judge_completeness": mean(judge_completeness),
        }

    @staticmethod
    def _evaluate_correctness(
        entry: dict,
        answer: str,
    ) -> float:
        """Evaluate generated answer against the golden-set reference answer."""

        if not entry.get("answerable", True):
            return 1.0

        reference_answer = entry.get("reference_answer", "").strip()

        if not reference_answer:
            return 0.0

        normalized_answer = answer.strip().casefold()
        normalized_reference = reference_answer.casefold()

        return float(
            normalized_reference in normalized_answer
        )

    @staticmethod
    def _evaluate_semantic_similarity(
        entry: dict,
        answer: str,
    ) -> float:
        """Measure semantic similarity between reference and generated answers."""

        reference_answer = entry.get("reference_answer", "").strip()
        generated_answer = answer.strip()

        if not reference_answer or not generated_answer:
            return 0.0

        reference_vector, answer_vector = embed_texts(
            [reference_answer, generated_answer]
        )

        return float(reference_vector @ answer_vector)

    @staticmethod
    def _evaluate_groundedness(
        entry: dict,
        answer: str,
        retrieved_results: list[dict],
    ) -> float:
        """Evaluate whether the answer is supported by retrieved context.

        The deterministic check uses the golden-set evidence statements
        as the reference for supported content. An answer is grounded
        when its substantive answer content is supported by the retrieved
        context.

        For unanswerable cases, groundedness follows abstention behavior.
        """

        if not entry.get("answerable", True):
            return GenerationEvaluator._evaluate_abstention(
                entry,
                answer,
            )

        evidence = [
            item.get("text", "").strip()
            for item in entry.get("evidence", [])
            if item.get("text")
        ]

        if not evidence:
            return 0.0

        retrieved_text = " ".join(
            result.get("content", "")
            for result in retrieved_results
            if result.get("content")
        )

        if not retrieved_text.strip() or not answer.strip():
            return 0.0

        answer_tokens = GenerationEvaluator._content_tokens(answer)
        context_tokens = GenerationEvaluator._content_tokens(retrieved_text)

        if not answer_tokens:
            return 0.0

        supported_tokens = answer_tokens & context_tokens

        return float(
            len(supported_tokens) / len(answer_tokens) >= 0.8
        )

    @staticmethod
    def _evaluate_citation(
        entry: dict,
        answer: str,
        retrieved_results: list[dict],
    ) -> float:
        """Evaluate citation/source compliance.

        For answerable cases, every expected source must be represented
        in the generated answer, and every cited expected source must
        have been retrieved.

        For unanswerable cases, citation compliance is satisfied when
        the answer does not cite any retrieved source.
        """

        answerable = entry.get("answerable", True)

        expected_sources = {
            source
            for source in entry.get("expected_sources", [])
            if source
        }

        retrieved_sources = {
            result.get("source")
            for result in retrieved_results
            if result.get("source")
        }

        if not answerable:
            return float(
                not GenerationEvaluator._contains_source_name(
                    answer,
                    retrieved_sources,
                )
            )

        if not expected_sources:
            return 0.0

        cited_expected_sources = {
            source
            for source in expected_sources
            if GenerationEvaluator._contains_source_name(
                answer,
                {source},
            )
        }

        if cited_expected_sources != expected_sources:
            return 0.0

        return float(
            cited_expected_sources <= retrieved_sources
        )

    @staticmethod
    def _evaluate_abstention(
        entry: dict,
        answer: str,
    ) -> float:
        """Evaluate whether an unanswerable case was appropriately declined."""

        answerable = entry.get("answerable", True)

        if answerable:
            return 1.0

        normalized = answer.strip().lower()

        if not normalized:
            return 1.0

        abstention_markers = (
            "i don't know",
            "i do not know",
            "i don't have enough information",
            "i do not have enough information",
            "cannot answer",
            "can't answer",
            "not enough information",
            "insufficient information",
            "no information",
            "unable to answer",
            "not provided",
            "not available",
        )

        return float(
            any(marker in normalized for marker in abstention_markers)
        )

    @staticmethod
    def _contains_source_name(
        answer: str,
        sources: set[str],
    ) -> bool:
        """Return whether any exact source name appears in the answer."""

        normalized_answer = answer.casefold()

        return any(
            source.casefold() in normalized_answer
            for source in sources
        )

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        """Return normalized content tokens for deterministic overlap checks."""

        return {
            token
            for token in text.casefold().split()
            if token.isalnum()
        }