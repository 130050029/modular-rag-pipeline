"""
test_docling_integration.py -- exercises the actual Docling extraction path
for real, the way test_postgres_integration.py and test_redis_integration.py
exercise their respective backends. test_extractors.py only tests the
error path (extraction raises correctly when Docling ISN'T configured) --
this file is the sole real evidence that Docling extraction itself works.

Skips (not fails) if docling isn't installed, so a normal `pytest tests/`
run is unaffected either way -- docling is a heavy, optional dependency.

NOTE ON SCOPE: extract_text() is a pure function -- no database, no vector
index, nothing persistent is touched here, so there's no cleanup step
needed (unlike the Postgres/Redis integration tests). This only verifies
extraction itself works; it does NOT verify table-to-Markdown conversion
specifically, since our synthetic sample PDF (data/manual_test_files/
sample_document.pdf) contains plain text only, no real table structure --
verifying that specifically requires a real PDF with an actual table,
which is a manual step (see manual_test_docling.py).
"""

import pytest


def _docling_available() -> bool:
    try:
        import docling  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _docling_available(),
    reason="docling not installed -- run `pip install docling` to enable this test "
           "(it's a heavy, optional dependency, not installed by default).",
)


def test_docling_extracts_text_from_pdf(monkeypatch):
    monkeypatch.setattr("config.PDF_EXTRACTION_METHOD", "docling")
    from rag.ingestion.extractors import extract_text

    with open("data/manual_test_files/sample_document.pdf", "rb") as f:
        raw = f.read()

    text = extract_text("sample_document.pdf", raw)

    assert len(text) > 0
    assert "Renewable Energy Report" in text or "wind" in text.lower()


def test_docling_extracts_table_as_markdown_from_image(monkeypatch):
    """Verifies the actual point of using Docling over PyMuPDF: a table
    embedded in an image comes out as real Markdown pipe syntax, which our
    existing table-detection logic (rag/ingestion/tables.py) then picks up
    automatically with no extra code. Manually confirmed working by hand
    before this test was added -- see data/manual_test_files/sample_table.png."""
    import os

    path = "data/manual_test_files/sample_table.png"
    if not os.path.exists(path):
        pytest.skip(f"{path} not present -- add a PNG containing a table to enable this test.")

    monkeypatch.setattr("config.PDF_EXTRACTION_METHOD", "docling")
    from rag.ingestion.extractors import extract_text

    with open(path, "rb") as f:
        raw = f.read()

    text = extract_text("sample_table.png", raw)

    assert len(text) > 0
    assert "|" in text, "Expected Markdown table pipe syntax in Docling's output"
