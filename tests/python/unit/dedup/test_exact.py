import hashlib

from rag.dedup.exact import hash_text


def test_hash_is_deterministic():
    assert hash_text("hello world") == hash_text("hello world")


def test_hash_differs_for_different_text():
    assert hash_text("hello") != hash_text("world")


def test_hash_matches_sha256():
    text = "hello world"

    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()

    assert hash_text(text) == expected


def test_hash_handles_empty_text():
    expected = hashlib.sha256(b"").hexdigest()

    assert hash_text("") == expected


def test_hash_handles_unicode():
    text = "café — résumé 日本語"

    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()

    assert hash_text(text) == expected


def test_hash_is_sensitive_to_whitespace():
    assert hash_text("hello world") != hash_text("hello  world")
    assert hash_text("hello world") != hash_text("hello world ")