"""
NOTE: fake_embeddings are deterministic but NOT semantically meaningful --
they don't encode "this text is about France" the way a real model would.
So this test only verifies the retrieval MECHANICS (embed query -> search
-> fetch parent content) work correctly, not that retrieval finds the most
semantically relevant chunk among several -- that needs the real model
(a slower, separate integration test, not included here).
"""

from rag.ingestion.pipeline import ingest_document
from rag.retrieval import retrieve


def test_retrieve_returns_parent_content(fake_embeddings):
    ingest_document("doc.txt", "Paris is the capital of France and a major European city with many landmarks.")

    results = retrieve("What is the capital of France?", top_k=1)

    assert len(results) == 1
    assert "Paris" in results[0]["content"]
    assert results[0]["source"] == "doc.txt"


def test_retrieve_with_empty_index_returns_nothing(fake_embeddings):
    assert retrieve("anything at all") == []
