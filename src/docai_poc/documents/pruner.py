"""Prune and chunk documents ahead of extraction.

Pruning drops pages that add nothing to entity extraction (explicit
boilerplate pages, or blank pages on the native/text side of a document)
before the remaining pages are grouped into chunks sized to fit Document
AI's synchronous per-request page limit. Chunking only ever groups
*kept* native-side pages for repeated sync calls; scanned pages are
handled by whichever method (`ExtractionMethod`) the caller selects, since
we cannot safely judge scanned-page content without running OCR first.
"""

from __future__ import annotations

import io

from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader, PdfWriter

from docai_poc.documents.classifier import DocumentClassification
from docai_poc.documents.loader import LoadedPdf

DEFAULT_BLANK_PAGE_MAX_CHARS = 3
DEFAULT_MAX_PAGES_PER_CHUNK = 15


class PruneConfig(BaseModel):
    """Configuration for `prune`.

    Attributes:
        drop_pages: 1-indexed pages to always drop (e.g. known cover/TOC
            pages), regardless of content.
        drop_blank_native_pages: If `True`, native (text-layer) pages whose
            text is at or below `blank_page_max_chars` are dropped. Never
            applied to pages classified as scanned, since an empty text
            layer there just means OCR hasn't run yet, not that the page
            is empty.
        blank_page_max_chars: Threshold used by `drop_blank_native_pages`.
        max_pages_per_chunk: Maximum pages per chunk, sized to Document
            AI's synchronous per-request page limit.
    """

    model_config = ConfigDict(frozen=True)

    drop_pages: frozenset[int] = frozenset()
    drop_blank_native_pages: bool = True
    blank_page_max_chars: int = DEFAULT_BLANK_PAGE_MAX_CHARS
    max_pages_per_chunk: int = Field(default=DEFAULT_MAX_PAGES_PER_CHUNK, ge=1)


class PruneResult(BaseModel):
    """Outcome of pruning: which pages survive, and how they're chunked.

    Attributes:
        kept_pages: 1-indexed pages that survive pruning, in order.
        dropped_pages: 1-indexed pages that were dropped, in order.
        chunks: `kept_pages` grouped into runs no longer than
            `PruneConfig.max_pages_per_chunk`, preserving order.
    """

    model_config = ConfigDict(frozen=True)

    kept_pages: list[int]
    dropped_pages: list[int]
    chunks: list[list[int]]


def _chunk(pages: list[int], max_pages_per_chunk: int) -> list[list[int]]:
    return [pages[i : i + max_pages_per_chunk] for i in range(0, len(pages), max_pages_per_chunk)]


def prune(
    document: LoadedPdf,
    classification: DocumentClassification,
    config: PruneConfig | None = None,
) -> PruneResult:
    """Drop unneeded pages and chunk the remainder for extraction.

    Args:
        document: The loaded document to prune.
        classification: `classify_document` output for this same document,
            used to protect scanned pages from blank-page pruning.
        config: Pruning configuration; defaults to `PruneConfig()`.

    Returns:
        The kept/dropped pages and chunk groupings.
    """
    config = config or PruneConfig()
    dropped = set(config.drop_pages)

    if config.drop_blank_native_pages:
        for page_num, (text, is_native) in enumerate(
            zip(document.page_texts, classification.page_is_native, strict=True), start=1
        ):
            if is_native and len(text.strip()) <= config.blank_page_max_chars:
                dropped.add(page_num)

    kept = [p for p in range(1, document.page_count + 1) if p not in dropped]
    chunks = _chunk(kept, config.max_pages_per_chunk)
    return PruneResult(kept_pages=kept, dropped_pages=sorted(dropped), chunks=chunks)


def extract_chunk_bytes(document: LoadedPdf, page_numbers: list[int]) -> bytes:
    """Build a standalone PDF containing only the given 1-indexed pages.

    Args:
        document: The source document.
        page_numbers: 1-indexed page numbers to include, in the desired
            output order.

    Returns:
        The bytes of a new PDF containing exactly those pages.
    """
    reader = PdfReader(io.BytesIO(document.raw_bytes))
    writer = PdfWriter()
    for page_number in page_numbers:
        writer.add_page(reader.pages[page_number - 1])
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
