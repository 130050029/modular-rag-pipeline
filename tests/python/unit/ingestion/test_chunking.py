from rag.ingestion import chunking


def test_fixed_chunking_respects_size_and_overlap(monkeypatch):
    monkeypatch.setattr(chunking, "PARENT_CHUNK_SIZE_WORDS", 20)
    monkeypatch.setattr(chunking, "SMALL_CHUNK_SIZE_WORDS", 5)
    monkeypatch.setattr(chunking, "SMALL_CHUNK_OVERLAP_WORDS", 1)
    monkeypatch.setattr(chunking, "CHUNKING_STRATEGY", "fixed")

    text = " ".join(f"word{i}" for i in range(20))
    groups = chunking.chunk_document(text)

    assert len(groups) == 1
    assert groups[0]["parent_text"] == text
    assert len(groups[0]["small_texts"]) == 5
    assert groups[0]["small_texts"][0] == "word0 word1 word2 word3 word4"
    assert groups[0]["small_texts"][1].split()[0] == "word4"


def test_empty_text_produces_no_chunks():
    assert chunking.chunk_document("") == []


def test_fixed_chunking_splits_large_document(monkeypatch):
    monkeypatch.setattr(chunking, "PARENT_CHUNK_SIZE_WORDS", 5)
    monkeypatch.setattr(chunking, "SMALL_CHUNK_SIZE_WORDS", 3)
    monkeypatch.setattr(chunking, "SMALL_CHUNK_OVERLAP_WORDS", 1)
    monkeypatch.setattr(chunking, "CHUNKING_STRATEGY", "fixed")

    text = " ".join(f"word{i}" for i in range(10))
    groups = chunking.chunk_document(text)

    assert len(groups) == 2
    assert groups[0]["parent_text"] == "word0 word1 word2 word3 word4"
    assert groups[1]["parent_text"] == "word5 word6 word7 word8 word9"


def test_table_becomes_single_chunk(monkeypatch):
    monkeypatch.setattr(chunking, "PARENT_CHUNK_SIZE_WORDS", 3)
    monkeypatch.setattr(chunking, "SMALL_CHUNK_SIZE_WORDS", 2)
    monkeypatch.setattr(chunking, "SMALL_CHUNK_OVERLAP_WORDS", 1)

    table = "| Name | Price |\n|------|-------|\n| A | $10 |"
    groups = chunking.chunk_document(table)

    assert len(groups) == 1
    assert groups[0]["parent_text"] == table
    assert groups[0]["small_texts"] == [table]


def test_prose_around_table_preserves_document_order(monkeypatch):
    monkeypatch.setattr(chunking, "PARENT_CHUNK_SIZE_WORDS", 100)
    monkeypatch.setattr(chunking, "SMALL_CHUNK_SIZE_WORDS", 100)

    table = "| Name | Price |\n|------|-------|\n| A | $10 |"
    text = f"Before the table.\n\n{table}\n\nAfter the table."

    groups = chunking.chunk_document(text)

    assert len(groups) == 3
    assert groups[0]["parent_text"] == "Before the table."
    assert groups[1]["parent_text"] == table
    assert groups[2]["parent_text"] == "After the table."


def test_semantic_strategy_uses_sentence_similarity(monkeypatch):
    monkeypatch.setattr(chunking, "CHUNKING_STRATEGY", "semantic")
    monkeypatch.setattr(chunking, "SEMANTIC_CHUNK_SIMILARITY_DROP", 0.2)
    monkeypatch.setattr(chunking, "SEMANTIC_CHUNK_MAX_WORDS", 100)

    def fake_embed_texts(sentences):
        import numpy as np

        vectors = [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
        ]
        return np.array(vectors)

    monkeypatch.setattr("rag.embeddings.embed_texts", fake_embed_texts)

    words = (
        "Cats are common pets. "
        "Cats are popular household animals. "
        "Quantum mechanics studies particles."
    ).split()

    chunks = chunking._split_semantic(words)

    assert len(chunks) == 2
    assert "Cats are common pets." in chunks[0]
    assert "Cats are popular household animals." in chunks[0]
    assert "Quantum mechanics studies particles." in chunks[1]


def test_semantic_strategy_respects_max_word_limit(monkeypatch):
    monkeypatch.setattr(chunking, "CHUNKING_STRATEGY", "semantic")
    monkeypatch.setattr(chunking, "SEMANTIC_CHUNK_SIMILARITY_DROP", 1.0)
    monkeypatch.setattr(chunking, "SEMANTIC_CHUNK_MAX_WORDS", 6)

    def fake_embed_texts(sentences):
        import numpy as np

        return np.ones((len(sentences), 2))

    monkeypatch.setattr("rag.embeddings.embed_texts", fake_embed_texts)

    words = (
        "One two three four. "
        "Five six seven eight. "
        "Nine ten."
    ).split()

    chunks = chunking._split_semantic(words)

    assert len(chunks) == 2
    assert len(chunks[0].split()) <= 6