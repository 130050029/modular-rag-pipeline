"""Query decomposition for Phase B query intelligence."""
from typing import Protocol

import requests
import config


def build_decomposition_prompt(query: str) -> str:
    """Build a prompt that asks an LLM to split a complex query."""
    return (
        "Break the user's question into the smallest useful set of "
        "independent questions needed to retrieve the information required "
        "to answer it. "
        "Preserve important entities, names, dates, numbers, and constraints. "
        "Each question must be understandable on its own. "
        "Return one question per line, with no numbering, bullets, "
        "explanations, or other text.\n\n"
        f"User question: {query}\n\n"
        "Decomposed questions:"
    )


class QueryDecomposer(Protocol):
    """Interface for decomposing a user query."""

    def decompose(self, query: str) -> list[str]:
        """Return independent retrieval questions derived from the query."""
        raise NotImplementedError


class OllamaQueryDecomposer(QueryDecomposer):
    """Query decomposer backed by a local Ollama model."""

    def decompose(self, query: str) -> list[str]:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": build_decomposition_prompt(query),
                "stream": False,
                "options": {
                    "num_predict": config.LLM_MAX_TOKENS,
                },
            },
            timeout=config.LLM_TIMEOUT,
        )
        response.raise_for_status()

        raw_response = response.json()["response"]

        queries = [
            line.strip()
            for line in raw_response.splitlines()
            if line.strip()
        ]

        if not queries:
            raise RuntimeError(
                "Query decomposer returned an empty response"
            )

        max_queries = config.QUERY_DECOMPOSITION_MAX_QUERIES

        return queries[:max_queries]