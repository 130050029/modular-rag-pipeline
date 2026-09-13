import json

import pytest

from rag.evaluation.nvidia_llm_judge import NvidiaLLMJudge
import config

class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_build_prompt_contains_question_answer_and_context():
    prompt = NvidiaLLMJudge._build_prompt(
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

    assert "What was revenue?" in prompt
    assert "Revenue was 160000." in prompt
    assert "[Source: table.txt]" in prompt


def test_parse_valid_json_response():
    content = json.dumps(
        {
            "correctness": 1.0,
            "groundedness": 0.9,
            "relevance": 1.0,
            "completeness": 0.8,
            "reasoning": "The answer is supported by the context.",
        }
    )

    result = NvidiaLLMJudge._parse_result(content)

    assert result.correctness == 1.0
    assert result.groundedness == 0.9
    assert result.relevance == 1.0
    assert result.completeness == 0.8
    assert result.reasoning == "The answer is supported by the context."


def test_parse_json_inside_code_fence():
    content = """```json
{
  "correctness": 1.0,
  "groundedness": 1.0,
  "relevance": 1.0,
  "completeness": 1.0,
  "reasoning": "Correct."
}
```"""

    result = NvidiaLLMJudge._parse_result(content)

    assert result.correctness == 1.0
    assert result.groundedness == 1.0


def test_parse_rejects_invalid_json():
    with pytest.raises(ValueError, match="invalid JSON"):
        NvidiaLLMJudge._parse_result(
            "This is not JSON."
        )


def test_parse_rejects_missing_fields():
    content = json.dumps(
        {
            "correctness": 1.0,
            "groundedness": 1.0,
        }
    )

    with pytest.raises(
        ValueError,
        match="missing fields",
    ):
        NvidiaLLMJudge._parse_result(content)


def test_parse_rejects_score_outside_range():
    content = json.dumps(
        {
            "correctness": 1.5,
            "groundedness": 1.0,
            "relevance": 1.0,
            "completeness": 1.0,
            "reasoning": "Invalid score.",
        }
    )

    with pytest.raises(
        ValueError,
        match="must be between",
    ):
        NvidiaLLMJudge._parse_result(content)


def test_evaluate_requires_api_key(monkeypatch):
    monkeypatch.setattr(
        config,
        "NVIDIA_API_KEY",
        None,
    )

    judge = NvidiaLLMJudge()

    with pytest.raises(
        RuntimeError,
        match="NVIDIA_API_KEY",
    ):
        judge.evaluate(
            query="What was revenue?",
            answer="Revenue was 160000.",
            context=[],
            reference_answer="Revenue was 160000.",
        )

def test_evaluate_parses_nvidia_response(monkeypatch):
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "correctness": 1.0,
                            "groundedness": 1.0,
                            "relevance": 1.0,
                            "completeness": 0.9,
                            "reasoning": "The answer is well supported.",
                        }
                    )
                }
            }
        ]
    }

    def fake_post(*args, **kwargs):
        assert args[0].endswith(
            "/v1/chat/completions"
        )

        assert kwargs["json"]["model"] == (
            "meta/llama-3.2-3b-instruct"
        )

        assert kwargs["json"]["temperature"] == 0.0
        assert kwargs["json"]["stream"] is False

        return FakeResponse(payload)

    monkeypatch.setenv(
        "NVIDIA_API_KEY",
        "test-key",
    )

    monkeypatch.setattr(
        "rag.evaluation.nvidia_llm_judge.requests.post",
        fake_post,
    )

    judge = NvidiaLLMJudge(
        model="meta/llama-3.2-3b-instruct",
    )

    result = judge.evaluate(
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

    assert result.correctness == 1.0
    assert result.groundedness == 1.0
    assert result.relevance == 1.0
    assert result.completeness == 0.9
