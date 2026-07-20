"""
chunking.py -- text-splitting strategy. Two strategies now, chosen by
config.CHUNKING_STRATEGY:

  "fixed"    (default) -- fixed-size word windows with overlap.
  "semantic" -- splits where consecutive sentences' embedding similarity
                drops sharply (a proxy for "topic shift"), instead of at an
                arbitrary word count.

Both still produce the same output shape: a list of parent groups, each
with its own small chunks -- "small-to-retrieve, large-to-generate" is
enforced the same way regardless of which strategy split the text.
"""

import re
from config import (
    PARENT_CHUNK_SIZE_WORDS,
    SMALL_CHUNK_SIZE_WORDS,
    SMALL_CHUNK_OVERLAP_WORDS,
    CHUNKING_STRATEGY,
    SEMANTIC_CHUNK_SIMILARITY_DROP,
    SEMANTIC_CHUNK_MAX_WORDS,
)


def _split_fixed(words: list[str], size: int, overlap: int) -> list[str]:
    pieces = []
    start = 0
    while start < len(words):
        end = start + size
        piece = " ".join(words[start:end])
        if piece.strip():
            pieces.append(piece)
        if end >= len(words):
            break
        start = end - overlap
    return pieces


def _split_sentences(text: str) -> list[str]:
    # Simple sentence splitter -- good enough for a toy project; a real
    # pipeline would use a proper sentence tokenizer (e.g. nltk/spacy).
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


def _split_semantic(words: list[str]) -> list[str]:
    """Groups sentences into chunks, starting a new chunk wherever
    consecutive-sentence embedding similarity drops sharply."""
    from rag.embeddings import embed_texts   # imported lazily to avoid a hard
                                           # dependency for callers only using fixed chunking
    import numpy as np

    text = " ".join(words)
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text] if text.strip() else []

    vectors = embed_texts(sentences)   # already normalized -> dot product = cosine similarity

    chunks = []
    current = [sentences[0]]
    current_word_count = len(sentences[0].split())

    for i in range(1, len(sentences)):
        similarity = float(np.dot(vectors[i - 1], vectors[i]))
        similarity_drop = 1 - similarity
        would_exceed_cap = current_word_count + len(sentences[i].split()) > SEMANTIC_CHUNK_MAX_WORDS

        if similarity_drop >= SEMANTIC_CHUNK_SIMILARITY_DROP or would_exceed_cap:
            chunks.append(" ".join(current))
            current = [sentences[i]]
            current_word_count = len(sentences[i].split())
        else:
            current.append(sentences[i])
            current_word_count += len(sentences[i].split())

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(text: str) -> list[dict]:
    """Returns: [{"parent_text": str, "small_texts": [str, ...]}, ...]"""
    words = text.split()
    parent_texts = _split_fixed(words, PARENT_CHUNK_SIZE_WORDS, overlap=0)

    groups = []
    for parent_text in parent_texts:
        parent_words = parent_text.split()

        if CHUNKING_STRATEGY == "semantic":
            small_texts = _split_semantic(parent_words)
        else:
            small_texts = _split_fixed(parent_words, SMALL_CHUNK_SIZE_WORDS, SMALL_CHUNK_OVERLAP_WORDS)

        groups.append({"parent_text": parent_text, "small_texts": small_texts})
    return groups