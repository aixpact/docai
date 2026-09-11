"""Classify a PDF (or its pages) as native text vs. scanned/image content.

The classification drives which Document AI processor and extraction
method (native vs. OCR, sync vs. batch) the pipeline picks — native pages
can skip OCR entirely, which is both faster and cheaper.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from docai_poc.documents.loader import LoadedPdf

DEFAULT_MIN_CHARS_PER_PAGE = 20
"""Pages with fewer extracted characters than this are treated as scanned."""


class DocumentKind(str, Enum):
    """Overall classification of a document's pages."""

    NATIVE = "native"
    SCANNED = "scanned"
    MIXED = "mixed"


class DocumentClassification(BaseModel):
    """Per-page and overall native-vs-scanned classification.

    Attributes:
        page_is_native: One flag per page (1:1 with `LoadedPdf.page_texts`),
            `True` if the page has a usable text layer.
        kind: Overall document classification derived from `page_is_native`.
    """

    model_config = ConfigDict(frozen=True)

    page_is_native: list[bool]
    kind: DocumentKind

    @property
    def native_page_ratio(self) -> float:
        """Fraction of pages classified as native (text-layer) pages."""
        if not self.page_is_native:
            return 0.0
        return sum(self.page_is_native) / len(self.page_is_native)


def classify_document(
    document: LoadedPdf,
    min_chars_per_page: int = DEFAULT_MIN_CHARS_PER_PAGE,
) -> DocumentClassification:
    """Classify each page of a document as native or scanned, and overall.

    Args:
        document: The loaded PDF to classify.
        min_chars_per_page: Minimum extracted (stripped) characters for a
            page to count as native; below this it's treated as scanned.

    Returns:
        The per-page and overall classification.
    """
    page_is_native = [len(text.strip()) >= min_chars_per_page for text in document.page_texts]

    if not page_is_native:
        kind = DocumentKind.SCANNED
    elif all(page_is_native):
        kind = DocumentKind.NATIVE
    elif not any(page_is_native):
        kind = DocumentKind.SCANNED
    else:
        kind = DocumentKind.MIXED

    return DocumentClassification(page_is_native=page_is_native, kind=kind)
