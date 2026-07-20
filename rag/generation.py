"""
generation.py -- builds the RAG prompt and calls Claude. Isolated so the
prompt template, context-assembly strategy, or model choice can change
without touching retrieval or server code.
"""

import requests
from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL


def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['content']}" for c in retrieved_chunks
    )
    return (
        "Answer the question using ONLY the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )


def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    if not ANTHROPIC_API_KEY:
        return (
            "[No ANTHROPIC_API_KEY set -- returning raw retrieved context instead]\n\n"
            + "\n---\n".join(c["content"] for c in retrieved_chunks)
        )

    prompt = build_prompt(query, retrieved_chunks)

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": ANTHROPIC_MODEL,
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["content"][0]["text"]