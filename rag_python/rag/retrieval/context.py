"""Build the context passed from retrieval to the generation layer."""

from config import CONTEXT_MAX_CHARS


def build_context(results: list[dict], max_chars: int = CONTEXT_MAX_CHARS) -> str:
    """Deduplicate and pack retrieved chunks within the context budget."""
    if max_chars <= 0:
        return ""

    seen = set()
    chunks = []
    total = 0

    for result in results:
        content = result.get("content", "").strip()

        if not content:
            continue

        key = content

        if key in seen:
            continue

        if total + len(content) > max_chars:
            break

        seen.add(key)
        chunks.append(content)
        total += len(content)

    return "\n\n".join(chunks)