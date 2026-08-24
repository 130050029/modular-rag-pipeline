from rag.dedup.exact import hash_text


def test_hash_is_deterministic():
    assert hash_text("hello world") == hash_text("hello world")


def test_hash_differs_for_different_text():
    assert hash_text("hello") != hash_text("world")
