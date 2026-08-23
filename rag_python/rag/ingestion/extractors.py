"""
extractors.py -- converts raw uploaded bytes into plain text, dispatched by
file extension via a registry. Add a new file type by writing one function
and adding one line to EXTRACTORS -- nothing else in the pipeline changes.

PDF (and, with Docling, image) extraction has two selectable methods, via
config.PDF_EXTRACTION_METHOD:
  "pymupdf" (default) -- fast, lightweight, raw text only.
  "docling" -- IBM's open-source (MIT-licensed) document intelligence
    toolkit. Outputs Markdown with tables preserved as real Markdown
    tables, so they're automatically picked up by our existing table
    detection (rag/ingestion/tables.py) with no extra code. Also the only
    option here that supports images (via OCR). Meaningfully heavier: pulls
    in real ML models as dependencies (layout detection, table structure
    recognition, OCR), so it's an opt-in upgrade, not the default.

NOTE: the docling path below has been implemented carefully against its
documented API but has NOT been run end-to-end in this environment (it
pulls in torch and several ML models -- too large to install in the
sandbox this was developed in). Test it directly before relying on it.
"""

import os


def _extract_txt(raw: bytes) -> str:
    return raw.decode("utf-8", errors="ignore")


def _extract_html(raw: bytes) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise RuntimeError("Install beautifulsoup4 to handle .html uploads: pip install beautifulsoup4")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _extract_pdf_pymupdf(raw: bytes) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError("Install PyMuPDF to handle .pdf uploads: pip install pymupdf")
    text_parts = []
    with fitz.open(stream=raw, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def _extract_with_docling(raw: bytes, filename: str) -> str:
    """Shared by PDF and image extraction when PDF_EXTRACTION_METHOD ==
    "docling" -- Docling's DocumentConverter handles both, dispatching
    internally based on the file's content/extension."""
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import DocumentStream
        from io import BytesIO
    except ImportError:
        raise RuntimeError(
            "Install docling to use PDF_EXTRACTION_METHOD='docling': pip install docling "
            "(this is a heavy dependency -- pulls in torch and several ML models)"
        )

    converter = DocumentConverter()
    source = DocumentStream(name=filename, stream=BytesIO(raw))
    result = converter.convert(source)
    return result.document.export_to_markdown()


def _extract_pdf(raw: bytes, filename: str) -> str:
    from config import PDF_EXTRACTION_METHOD   # read fresh, not at module top -- see near.py/pipeline.py for why
    if PDF_EXTRACTION_METHOD == "docling":
        return _extract_with_docling(raw, filename)
    return _extract_pdf_pymupdf(raw)


def _extract_image(raw: bytes, filename: str) -> str:
    from config import PDF_EXTRACTION_METHOD   # read fresh, not at module top
    if PDF_EXTRACTION_METHOD != "docling":
        raise RuntimeError(
            "Image extraction requires PDF_EXTRACTION_METHOD='docling' (PyMuPDF handles "
            "PDFs only, not images) -- set the environment variable and install docling."
        )
    return _extract_with_docling(raw, filename)


EXTRACTORS = {
    "txt": lambda raw, filename=None: _extract_txt(raw),
    "md": lambda raw, filename=None: _extract_txt(raw),
    "html": lambda raw, filename=None: _extract_html(raw),
    "htm": lambda raw, filename=None: _extract_html(raw),
    "pdf": _extract_pdf,
    "png": _extract_image,
    "jpg": _extract_image,
    "jpeg": _extract_image,
}


def extract_text(filename: str, raw: bytes) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
    extractor = EXTRACTORS.get(ext)
    if extractor is None:
        raise ValueError(f"No extractor registered for .{ext} files. Supported: {sorted(EXTRACTORS)}")
    return extractor(raw, filename)