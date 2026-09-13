import pytest

from rag.evaluation.llm_judge import LLMJudge, LLMJudgeResult


def test_llm_judge_result_has_expected_metrics():
    result = LLMJudgeResult(
        correctness=1.0,
        groundedness=0.8,
        relevance=1.0,
        completeness=0.5,
        reasoning="Mostly correct.",
    )

    assert result.correctness == 1.0
    assert result.groundedness == 0.8
    assert result.relevance == 1.0
    assert result.completeness == 0.5
    assert result.reasoning == "Mostly correct."


def test_llm_judge_result_is_immutable():
    result = LLMJudgeResult(
        correctness=1.0,
        groundedness=1.0,
        relevance=1.0,
        completeness=1.0,
    )

    with pytest.raises(AttributeError):
        result.correctness = 0.0


def test_llm_judge_evaluate_defines_contract():
    judge = LLMJudge()

    with pytest.raises(NotImplementedError):
        judge.evaluate(
            query="What was revenue?",
            answer="Revenue was 160000.",
            context=[
                {
                    "source": "table.txt",
                    "content": "Revenue was 160000.",
                }
            ],
            reference_answer="Revenue was 160000.",
        )