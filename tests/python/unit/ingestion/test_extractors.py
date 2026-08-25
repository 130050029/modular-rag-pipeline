import pytest

from rag.ingestion.extractors import extract_text


def test_txt_extraction():
    assert extract_text("notes.txt", b"hello world") == "hello world"


def test_markdown_extraction():
    assert extract_text("notes.md", b"# Heading\n\nSome text") == "# Heading\n\nSome text"


def test_html_extraction_strips_unwanted_tags():
    html = b"""
    <html>
      <body>
        <nav>Navigation</nav>
        <script>ignore()</script>
        <style>.hidden { display: none; }</style>
        <p>Real content</p>
        <footer>Footer</footer>
      </body>
    </html>
    """

    text = extract_text("page.html", html)

    assert "Real content" in text
    assert "ignore()" not in text
    assert "Navigation" not in text
    assert "Footer" not in text


def test_htm_uses_html_extractor():
    html = b"<html><body><p>Hello</p></body></html>"

    assert "Hello" in extract_text("page.htm", html)


def test_pdf_extraction_uses_pymupdf(monkeypatch):
    called = {}

    def fake_pdf_extractor(raw):
        called["raw"] = raw
        return "PDF content"

    monkeypatch.setattr(
        "rag.ingestion.extractors._extract_pdf_pymupdf",
        fake_pdf_extractor,
    )
    monkeypatch.setattr("config.PDF_EXTRACTION_METHOD", "pymupdf")

    assert extract_text("document.pdf", b"fake pdf") == "PDF content"
    assert called["raw"] == b"fake pdf"


def test_image_requires_docling(monkeypatch):
    monkeypatch.setattr("config.PDF_EXTRACTION_METHOD", "pymupdf")

    with pytest.raises(RuntimeError, match="docling"):
        extract_text("photo.png", b"fake image bytes")


def test_unsupported_extension_raises():
    with pytest.raises(ValueError, match="No extractor registered"):
        extract_text("data.xyz", b"whatever")


def test_extension_matching_is_case_insensitive():
    assert extract_text("NOTES.TXT", b"hello") == "hello"