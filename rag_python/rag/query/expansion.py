from typing import Protocol

import requests
import config

class QueryExpander(Protocol):
    def expand(self, query: str) -> list[str]:
        """Return alternative retrieval queries for the user's query."""
        ...

class OllamaQueryExpander:
    """Query expander backed by a local Ollama model."""

    def expand(self, query: str) -> list[str]:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": build_expansion_prompt(query),
                "stream": False,
                "options": {"num_predict": config.LLM_MAX_TOKENS},
            },
            timeout=config.LLM_TIMEOUT,
        )
        response.raise_for_status()

        raw = response.json()["response"].strip()

        if not raw:
            raise RuntimeError("Query expander returned an empty response")

        queries = [
            line.strip()
            for line in raw.splitlines()
            if line.strip()
        ]

        if not queries:
            raise RuntimeError("Query expander returned no queries")

        return queries

def build_expansion_prompt(query: str) -> str:
    """Build a prompt that asks an LLM for alternative retrieval queries."""
    return (
        "Generate 3 alternative search queries for retrieving documents "
        "that can answer the user's question. "
        "Use different wording or terminology while preserving the original "
        "meaning. Preserve important entities, names, dates, and numbers. "
        "Return exactly one query per line and no explanations.\n\n"
        f"User question: {query}\n\n"
        "Alternative queries:"
    )