"""LLM generation behind a small provider interface."""

from typing import Protocol

import requests
import config as config


class Generator(Protocol):
    def generate(self, query: str, chunks: list[dict]) -> str:
        ...


def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['content']}" for c in chunks
    )

    return (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


class OllamaGenerator:
    def generate(self, query: str, chunks: list[dict]) -> str:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": build_prompt(query, chunks),
                "stream": False,
                "options": {"num_predict": config.LLM_MAX_TOKENS},
            },
            timeout=config.LLM_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["response"]


class AnthropicGenerator:
    def generate(self, query: str, chunks: list[dict]) -> str:
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": config.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": config.ANTHROPIC_MODEL,
                "max_tokens": config.LLM_MAX_TOKENS,
                "messages": [
                    {"role": "user", "content": build_prompt(query, chunks)}
                ],
            },
            timeout=config.LLM_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]


def get_generator(provider: str | None = None) -> Generator:
    provider = provider or config.LLM_PROVIDER

    if provider == "ollama":
        return OllamaGenerator()

    if provider == "anthropic":
        return AnthropicGenerator()

    raise ValueError(f"Unknown LLM provider: {provider}")


def generate_answer(query: str, chunks: list[dict]) -> str:
    return get_generator().generate(query, chunks)