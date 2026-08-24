from rag.ingestion import chunking


def test_fixed_chunking_respects_size_and_overlap(monkeypatch):
    monkeypatch.setattr(chunking, "PARENT_CHUNK_SIZE_WORDS", 20)
    monkeypatch.setattr(chunking, "SMALL_CHUNK_SIZE_WORDS", 5)
    monkeypatch.setattr(chunking, "SMALL_CHUNK_OVERLAP_WORDS", 1)
    monkeypatch.setattr(chunking, "CHUNKING_STRATEGY", "fixed")

    text = " ".join(f"word{i}" for i in range(20))
    groups = chunking.chunk_document(text)

    assert len(groups) == 1

    small = groups[0]["small_texts"]
    assert len(small) == 5
    assert small[0].split() == ["word0", "word1", "word2", "word3", "word4"]
    assert small[1].split()[0] == "word4"


def test_empty_text_produces_no_chunks():
    assert chunking.chunk_document("") == []