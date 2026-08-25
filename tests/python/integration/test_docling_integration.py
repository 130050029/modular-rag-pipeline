from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "manual_test_files"


def _docling_available():
    try:
        import docling  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _docling_available(),
    reason="Docling is not installed.",
)


def test_docling_extracts_pdf(monkeypatch):
    monkeypatch.setattr("config.PDF_EXTRACTION_METHOD", "docling")

    from rag.ingestion.extractors import extract_text

    path = DATA_DIR / "sample_document.pdf"
    text = extract_text(path.name, path.read_bytes())

    assert text.strip()
    assert "Renewable Energy Report" in text or "wind" in text.lower()


def test_docling_extracts_table_from_image(monkeypatch):
    path = DATA_DIR / "sample_table.png"

    if not path.exists():
        pytest.skip("sample_table.png is not available.")

    monkeypatch.setattr("config.PDF_EXTRACTION_METHOD", "docling")

    from rag.ingestion.extractors import extract_text

    text = extract_text(path.name, path.read_bytes())

    assert text.strip()
    assert "|" in text