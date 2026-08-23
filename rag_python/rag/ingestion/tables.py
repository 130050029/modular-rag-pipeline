"""
tables.py -- detects Markdown-style tables inside a chunk and produces a
separate, natural-language EMBEDDING description of it -- "embed a
translation, not the raw thing".

The raw table (as Markdown) still gets stored as `content` and shown to the
LLM at generation time -- only what gets EMBEDDED differs.

Two ways to generate that description (config.TABLE_EMBEDDING_DESCRIPTION):
  - "template" : cheap, deterministic -- column names + row count. No LLM call.
  - "llm"      : ask Claude for a one-sentence description -- higher quality,
                 costs one generation call per table detected.
Both are genuinely used in practice; template is the sensible default,
LLM-generated descriptions are an upgrade when retrieval quality on tables
specifically isn't good enough.

NOTE: this is intentionally lightweight -- it detects tables that are ALREADY
Markdown-formatted within a chunk (e.g. from an already-clean source, or
after a real extraction pipeline reconstructed one). It does NOT do the
page-spanning multi-page table reconstruction we discussed as a deeper,
more specialized problem -- that would live further upstream, in extractors.py,
before chunking ever sees the text.
"""

import re
from config import TABLE_EMBEDDING_DESCRIPTION

_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


def is_table_like(text: str) -> bool:
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    table_lines = sum(1 for l in lines if _TABLE_ROW_RE.match(l))
    return table_lines >= max(2, len(lines) // 2)   # majority of lines look like table rows


def _parse_header_and_row_count(text: str) -> tuple[list[str], int]:
    lines = [l for l in text.splitlines() if _TABLE_ROW_RE.match(l)]
    header_cells = [c.strip() for c in lines[0].strip("|").split("|")] if lines else []
    # first line = header, second line is usually the '---|---' separator
    data_rows = max(0, len(lines) - 2)
    return header_cells, data_rows


def _template_description(text: str) -> str:
    columns, row_count = _parse_header_and_row_count(text)
    col_str = ", ".join(columns) if columns else "unknown columns"
    return f"A table with columns: {col_str}, containing {row_count} rows of data."


def _llm_description(text: str) -> str:
    import requests
    from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL

    if not ANTHROPIC_API_KEY:
        return _template_description(text)   # graceful fallback if no API key set

    prompt = f"In one short sentence, describe what this table contains (topic, not exact values):\n\n{text}"
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={"model": ANTHROPIC_MODEL, "max_tokens": 60, "messages": [{"role": "user", "content": prompt}]},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["content"][0]["text"].strip()


def split_into_segments(text: str) -> list[tuple[str, str]]:
    """Splits `text` into an ORDER-PRESERVING list of (segment_type, content)
    tuples, segment_type in {"table", "prose"} -- e.g. a document reading
    "intro, table, conclusion" produces [("prose", intro), ("table", ...),
    ("prose", conclusion)], not tables-first-then-everything-else.

    This replaces an earlier version (split_table_blocks) that extracted all
    table blocks first and lumped ALL remaining lines into one combined
    prose blob -- which silently merged unrelated prose appearing before and
    after a table into a single chunk, and always numbered tables as
    position 0 regardless of where they actually appeared in the document.

    Must still run on the RAW text, before any word-based chunking has a
    chance to collapse newlines -- table detection depends on line
    structure, which word-based chunking's `" ".join(words)` destroys.
    """
    lines = text.splitlines()
    segments: list[tuple[str, str]] = []
    buffer: list[str] = []
    buffer_type: str | None = None

    def flush():
        nonlocal buffer, buffer_type
        if buffer:
            content = "\n".join(buffer)
            # Re-verify with the majority-of-lines heuristic, not just
            # "every line in this run matched the single-row regex" --
            # protects against a lone stray "|" in prose being misread.
            final_type = "table" if (buffer_type == "table" and is_table_like(content)) else "prose"
            segments.append((final_type, content))
        buffer = []
        buffer_type = None

    for line in lines:
        line_type = "table" if _TABLE_ROW_RE.match(line) else "prose"
        if buffer_type is not None and line_type != buffer_type:
            flush()
        buffer_type = line_type
        buffer.append(line)
    flush()

    return segments


def get_embedding_text(chunk_text: str) -> str:
    """Returns what should actually be embedded for this chunk: a generated
    description if it looks like a table, otherwise the chunk's own text
    unchanged."""
    if not is_table_like(chunk_text):
        return chunk_text

    if TABLE_EMBEDDING_DESCRIPTION == "llm":
        return _llm_description(chunk_text)
    return _template_description(chunk_text)