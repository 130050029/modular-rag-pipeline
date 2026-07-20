"""
extractors.py -- converts raw uploaded bytes into plain text, dispatched by
file extension via a registry. Add a new file type by writing one function
and adding one line to EXTRACTORS -- nothing else in the pipeline changes.

Mirrors what tools like Unstructured.io/Docling do at a larger scale: one
format-detection step, then a format-specific parser, producing a single
plain-text representation the rest of the pipeline never has to think about
format for again.
"""


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


def _extract_pdf(raw: bytes) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError("Install PyMuPDF to handle .pdf uploads: pip install pymupdf")
    text_parts = []
    with fitz.open(stream=raw, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


EXTRACTORS = {
    "txt": _extract_txt,
    "md": _extract_txt,
    "html": _extract_html,
    "htm": _extract_html,
    "pdf": _extract_pdf,
}


def extract_text(filename: str, raw: bytes) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
    extractor = EXTRACTORS.get(ext)
    if extractor is None:
        raise ValueError(f"No extractor registered for .{ext} files. Supported: {sorted(EXTRACTORS)}")
    return extractor(raw)