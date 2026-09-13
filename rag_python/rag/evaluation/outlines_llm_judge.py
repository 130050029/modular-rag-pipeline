"""Local Outlines-backed LLM-as-a-judge implementation."""

from pydantic import BaseModel, Field
from typing import Literal
import config

from rag.evaluation.llm_judge import LLMJudge, LLMJudgeResult


class JudgeOutput(BaseModel):
    correctness: Literal[1, 2, 3, 4, 5]
    groundedness: Literal[1, 2, 3, 4, 5]
    relevance: Literal[1, 2, 3, 4, 5]
    completeness: Literal[1, 2, 3, 4, 5]


JUDGE_SYSTEM_PROMPT = """\
You are an evaluation judge for a retrieval-augmented generation system.

Evaluate the generated answer using only the supplied question, retrieved
context, and reference answer when one is provided.

Return scores between 0.0 and 1.0.

correctness:
How factually correct is the answer relative to the question and reference
answer?

groundedness:
Are the claims in the answer supported by the supplied retrieved context?
Do not reward information that comes from outside the context.

relevance:
Does the answer directly address the question without unnecessary
digressions?

completeness:
Does the answer cover the important parts of the question that can be
answered from the supplied context?

For an unanswerable question, do not penalize an answer merely because it
does not provide the requested fact. A correct refusal based on insufficient
context can receive a high score.

Do not use outside knowledge when judging groundedness.

For every dimension, return an integer from 1 to 5.

1 = completely poor
2 = mostly poor
3 = partially correct / mixed
4 = mostly strong
5 = completely strong

Return only the required structured result.
Do not generate decimal scores.
Do not generate explanations or reasoning.
"""


class OutlinesLLMJudge(LLMJudge):
    """Evaluate generated answers using a local open model and Outlines."""

    def __init__(
        self,
        model_name: str | None = None,
    ):
        self.model_name = (
            model_name
            or getattr(
                config,
                "OUTLINES_JUDGE_MODEL",
                "HuggingFaceTB/SmolLM2-135M-Instruct",
            )
        )

        self._model = None

    @property
    def model(self):
        """Load the model lazily so importing the evaluator stays cheap."""

        if self._model is None:
            self._model = self._load_model()

        return self._model

    def _load_model(self):
        import outlines
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
        )

        return outlines.from_transformers(
            hf_model,
            tokenizer,
        )

    @staticmethod
    def _normalize_score(score: int) -> float:
        return (score - 1) / 4

    def evaluate(
        self,
        query: str,
        answer: str,
        context: list[dict],
        reference_answer: str | None = None,
    ) -> LLMJudgeResult:

        prompt = self._build_prompt(
            query=query,
            answer=answer,
            context=context,
            reference_answer=reference_answer,
        )

        result = self.model(
            prompt,
            JudgeOutput,
            max_new_tokens=128,
        )

        data = self._normalize_result(result)

        return LLMJudgeResult(
            correctness=self._normalize_score(data.correctness),
            groundedness=self._normalize_score(data.groundedness),
            relevance=self._normalize_score(data.relevance),
            completeness=self._normalize_score(data.completeness),
        )

    @staticmethod
    def _normalize_result(result) -> JudgeOutput:
        """
        Normalize Outlines output into JudgeOutput.

        Different Outlines/model versions can return either a Pydantic
        instance, a dictionary, or a JSON string. Keep the normalization
        here so the rest of the evaluator always receives a validated
        JudgeOutput.

        Some small instruction models can also produce malformed numeric
        values despite the structured-generation constraint. Such output
        should fail clearly rather than silently producing misleading
        evaluation scores.
        """

        if isinstance(result, JudgeOutput):
            return result

        if isinstance(result, str):
            import json

            try:
                result = json.loads(result)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Outlines judge returned invalid JSON."
                ) from exc

        if isinstance(result, dict):
            return JudgeOutput.model_validate(result)

        try:
            return JudgeOutput.model_validate(result)
        except Exception as exc:
            raise ValueError(
                "Outlines judge returned an unexpected result type: "
                f"{type(result).__name__}"
            ) from exc

    @staticmethod
    def _build_prompt(
        query: str,
        answer: str,
        context: list[dict],
        reference_answer: str | None,
    ) -> str:

        context_parts = []

        for result in context:
            source = (
                result.get("source")
                or result.get("filename")
                or result.get("file_name")
                or "<unknown>"
            )

            content = result.get("content", "").strip()

            if not content:
                continue

            context_parts.append(
                f"[Source: {source}]\n{content}"
            )

        context_text = "\n\n".join(context_parts)

        reference_text = (
            reference_answer.strip()
            if reference_answer
            else "No reference answer provided."
        )

        return (
            f"{JUDGE_SYSTEM_PROMPT}\n\n"
            f"Question:\n{query}\n\n"
            f"Reference answer:\n{reference_text}\n\n"
            f"Retrieved context:\n{context_text}\n\n"
            f"Generated answer:\n{answer}\n\n"
        )