"""NVIDIA-backed LLM-as-a-judge implementation."""

import json
import os

import requests

import config

from rag.evaluation.llm_judge import LLMJudge, LLMJudgeResult


NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

DEFAULT_TIMEOUT = 180

JUDGE_SYSTEM_PROMPT = """\
You are an evaluation judge for a retrieval-augmented generation system.

Evaluate the generated answer using only the supplied question, context,
and reference answer when one is provided.

Return ONLY valid JSON with exactly these fields:

{
  "correctness": 0.0,
  "groundedness": 0.0,
  "relevance": 0.0,
  "completeness": 0.0,
  "reasoning": "brief explanation"
}

All four scores must be numbers between 0.0 and 1.0.

Scoring:

correctness:
How factually correct is the answer relative to the question and reference
answer?

groundedness:
Are the claims in the answer supported by the supplied context?
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

CRITICAL REQUIREMENT: Output raw JSON only. Do not wrap the JSON in markdown code blocks like ```json ... ```. No conversational text before or after.
"""


import json
import re


class NvidiaLLMJudge(LLMJudge):
    """Evaluate generated answers using an NVIDIA-hosted LLM."""

    def __init__(
        self,
        model: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.model = model or config.NVIDIA_JUDGE_MODEL
        self.timeout = timeout

    def evaluate(
        self,
        query: str,
        answer: str,
        context: list[dict],
        reference_answer: str | None = None,
    ) -> LLMJudgeResult:

        if not config.NVIDIA_API_KEY:
            raise RuntimeError(
                "NVIDIA_API_KEY environment variable is not set."
            )

        prompt = self._build_prompt(
            query=query,
            answer=answer,
            context=context,
            reference_answer=reference_answer,
        )

        response = requests.post(
            NVIDIA_API_URL,
            headers={
                "Authorization": f"Bearer {config.NVIDIA_API_KEY}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": JUDGE_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 512,
                "stream": False,
            },
            timeout=self.timeout,
        )

        response.raise_for_status()

        payload = response.json()

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "Unexpected NVIDIA API response format."
            ) from exc

        return self._parse_result(content)

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
            f"Question:\n{query}\n\n"
            f"Reference answer:\n{reference_text}\n\n"
            f"Retrieved context:\n{context_text}\n\n"
            f"Generated answer:\n{answer}\n\n"
            "Evaluate the generated answer according to the instructions "
            "and return only the required JSON object."
        )

    @staticmethod
    def _parse_result(content: str) -> LLMJudgeResult:
        cleaned = content.strip()

        # Handle models that wrap JSON in a code fence or add reasoning prefixes
        if "{" in cleaned and "}" in cleaned:
            # Extract everything between the first '{' and the last '}'
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            cleaned = cleaned[start:end]
        else:
            # Fallback to existing logic if braces aren't found cleanly
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            # Print or log the raw response so you can see EXACTLY what DeepSeek returned
            print(f"\n--- DEBUG: RAW MODEL OUTPUT FAILED TO PARSE --- \n{content}\n-----------------------------------------------\n")
            raise ValueError(
                f"NVIDIA judge returned invalid JSON. Raw content printed to stdout."
            ) from exc

        required_fields = (
            "correctness",
            "groundedness",
            "relevance",
            "completeness",
            "reasoning",
        )

        missing = [
            field
            for field in required_fields
            if field not in data
        ]

        if missing:
            raise ValueError(
                "NVIDIA judge response is missing fields: "
                + ", ".join(missing)
            )

        try:
            scores = {
                field: float(data[field])
                for field in (
                    "correctness",
                    "groundedness",
                    "relevance",
                    "completeness",
                )
            }
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "NVIDIA judge returned non-numeric values for scores."
            ) from exc

        for field, score in scores.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"NVIDIA judge score '{field}' must be between "
                    f"0.0 and 1.0, got {score}."
                )

        return LLMJudgeResult(
            correctness=scores["correctness"],
            groundedness=scores["groundedness"],
            relevance=scores["relevance"],
            completeness=scores["completeness"],
            reasoning=str(data["reasoning"]),
        )