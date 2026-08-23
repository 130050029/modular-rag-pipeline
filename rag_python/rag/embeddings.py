"""
embeddings.py -- wraps the embedding model so every other file just calls
embed_texts()/embed_query() without knowing which model or library is used
underneath. Swapping to a different embedding model later means editing
only this file.
"""

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' (first run downloads it)...")
_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str]):
    """Embed a batch of chunk texts. Returns normalized vectors (for cosine similarity via inner product)."""
    return _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def embed_query(query: str):
    """Embed a single query string."""
    return _model.encode([query], normalize_embeddings=True, show_progress_bar=False)