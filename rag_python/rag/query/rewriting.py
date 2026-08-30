from typing import Protocol

import requests
import config

class QueryRewriter(Protocol):
    def rewrite(self, query: str) -> str:
        """Rewrite a user query into a standalone retrieval query."""
        ...

def build_rewrite_prompt(query: str) -> str:
    """Build a prompt that asks an LLM for one standalone search query."""
    return (
        "Rewrite the user's question into one concise, standalone search query "
        "for retrieving relevant documents. "
        "Preserve the original meaning and important entities, names, dates, "
        "and numbers. "
        "Return only the rewritten query, with no explanation.\n\n"
        f"User question: {query}\n\n"
        "Rewritten query:"
    )


class OllamaQueryRewriter:
    """Query rewriter backed by a local Ollama model."""

    def rewrite(self, query: str) -> str:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": build_rewrite_prompt(query),
                "stream": False,
                "options": {"num_predict": config.LLM_MAX_TOKENS},
            },
            timeout=config.LLM_TIMEOUT,
        )
        response.raise_for_status()

        rewritten = response.json()["response"].strip()

        if not rewritten:
            raise RuntimeError("Query rewriter returned an empty response")

        return rewritten