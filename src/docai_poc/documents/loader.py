"""Open and read PDF documents ahead of classification/pruning/extraction."""

from __future__ import annotations

import io
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from pypdf import PdfReader


class LoadedPdf(BaseModel):
    """A PDF read into memory with per-page text already extracted.

    Attributes:
        source: Caller-supplied identifier (file path or logical name).
        raw_bytes: The original PDF bytes, kept for downstream Document AI
            calls (sync content upload or chunk re-assembly).
        page_texts: Extracted text for each page, in page order. An empty
            string means the page has no (or negligible) embedded text
            layer, e.g. a scanned image page.
    """

    model_config = ConfigDict(frozen=True)

    source: str
    raw_bytes: bytes
    page_texts: list[str]

    @property
    def page_count(self) -> int:
        """Total number of pages in the document."""
        return len(self.page_texts)


def load_pdf_bytes(data: bytes, source: str) -> LoadedPdf:
    """Read a PDF already in memory and extract per-page text.

    Args:
        data: Raw PDF file contents.
        source: Caller-supplied identifier for logging/reporting.

    Returns:
        The loaded document with per-page text extracted.
    """
    reader = PdfReader(io.BytesIO(data))
    page_texts = [page.extract_text() or "" for page in reader.pages]
    return LoadedPdf(source=source, raw_bytes=data, page_texts=page_texts)


def load_pdf(path: str | Path) -> LoadedPdf:
    """Read a PDF from disk and extract per-page text.

    Args:
        path: Path to the PDF file.

    Returns:
        The loaded document with per-page text extracted.
    """
    path = Path(path)
    return load_pdf_bytes(path.read_bytes(), source=str(path))
