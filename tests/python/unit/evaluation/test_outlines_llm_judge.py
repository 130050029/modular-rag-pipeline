from unittest.mock import MagicMock

import pytest

from rag.evaluation.llm_judge import LLMJudgeResult
from rag.evaluation.outlines_llm_judge import (
    JudgeOutput,
    OutlinesLLMJudge,
)


def test_judge_output_accepts_valid_scores():
    result = JudgeOutput(
        correctness=1,
        groundedness=3,
        relevance=2,
        completeness=5,
    )

    assert result.correctness == 1
    assert result.groundedness == 3
    assert result.relevance == 2
    assert result.completeness == 5


def test_judge_output_rejects_score_above_one():
    with pytest.raises(ValueError):
        JudgeOutput(
            correctness=0,
            groundedness=0.8,
            relevance=0.9,
            completeness=0.7,
        )


def test_judge_output_rejects_negative_score():
    with pytest.raises(ValueError):
        JudgeOutput(
            correctness=1.0,
            groundedness=-0.1,
            relevance=0.9,
            completeness=0.7,
        )


def test_build_prompt_contains_all_evaluation_inputs():
    judge = OutlinesLLMJudge(
        model_name="test-model",
    )

    prompt = judge._build_prompt(
        query="What is the revenue?",
        answer="Revenue was 160000.",
        context=[
            {
                "source": "financials.txt",
                "content": "Revenue was 160000 in Q4.",
            }
        ],
        reference_answer="Revenue was 160000.",
    )

    assert "What is the revenue?" in prompt
    assert "Revenue was 160000." in prompt
    assert "financials.txt" in prompt
    assert "Revenue was 160000 in Q4." in prompt


def test_build_prompt_handles_missing_reference():
    judge = OutlinesLLMJudge(
        model_name="test-model",
    )

    prompt = judge._build_prompt(
        query="What is the answer?",
        answer="I don't know.",
        context=[],
        reference_answer=None,
    )

    assert "No reference answer provided." in prompt


def test_evaluate_converts_structured_output_to_judge_result():
    judge = OutlinesLLMJudge(
        model_name="test-model",
    )

    mock_model = MagicMock()

    mock_model.return_value = JudgeOutput(
        correctness=3,
        groundedness=1,
        relevance=4,
        completeness=2,
    )

    judge._model = mock_model

    result = judge.evaluate(
        query="What is the revenue?",
        answer="Revenue was 160000.",
        context=[
            {
                "source": "financials.txt",
                "content": "Revenue was 160000 in Q4.",
            }
        ],
        reference_answer="Revenue was 160000.",
    )

    assert isinstance(result, LLMJudgeResult)

    assert result.correctness == 0.5
    assert result.groundedness == 0.0
    assert result.relevance == 0.75
    assert result.completeness == 0.25


def test_model_is_loaded_lazily(monkeypatch):
    judge = OutlinesLLMJudge(
        model_name="test-model",
    )

    loader = MagicMock(return_value="mock-model")

    monkeypatch.setattr(
        judge,
        "_load_model",
        loader,
    )

    assert judge._model is None

    assert judge.model == "mock-model"
    assert loader.call_count == 1

    # Second access should reuse the loaded model.
    assert judge.model == "mock-model"
    assert loader.call_count == 1