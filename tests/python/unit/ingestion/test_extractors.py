import pytest

import rag.config as config
from rag.ingestion.extractors import extract_text


def test_txt_extraction():
    assert extract_text("notes.txt", b"hello world") == "hello world"


def test_html_extraction_strips_tags():
    html = b"<html><body><script>ignore()</script><p>Real content</p></body></html>"
    text = extract_text("page.html", html)

    assert "Real content" in text
    assert "ignore()" not in text


def test_unsupported_extension_raises():
    with pytest.raises(ValueError):
        extract_text("data.xyz", b"whatever")


def test_image_without_docling_configured_raises_clear_error(monkeypatch):
    monkeypatch.setattr(config, "PDF_EXTRACTION_METHOD", "pymupdf")

    with pytest.raises(RuntimeError, match="docling"):
        extract_text("photo.png", b"fake image bytes")