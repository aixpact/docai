"""Tests for docai_poc.documents.pruner."""

from __future__ import annotations

import io

from pypdf import PdfReader

from docai_poc.documents.classifier import DocumentClassification, DocumentKind, classify_document
from docai_poc.documents.loader import LoadedPdf, load_pdf_bytes
from docai_poc.documents.pruner import PruneConfig, extract_chunk_bytes, prune


def test_prune_drops_explicit_pages() -> None:
    doc = LoadedPdf(
        source="doc.pdf", raw_bytes=b"%PDF-fake", page_texts=["a" * 50, "b" * 50, "c" * 50]
    )
    classification = classify_document(doc)
    result = prune(doc, classification, PruneConfig(drop_pages=frozenset({2})))
    assert result.kept_pages == [1, 3]
    assert result.dropped_pages == [2]


def test_prune_drops_blank_native_pages_but_not_scanned() -> None:
    doc = LoadedPdf(
        source="doc.pdf",
        raw_bytes=b"%PDF-fake",
        page_texts=["substantial native content", "", ""],
    )
    # Page 2 is a genuinely blank native page; page 3 we mark as scanned
    # (no text layer yet) to prove it is protected from blank-page pruning.
    classification = DocumentClassification(
        page_is_native=[True, True, False], kind=DocumentKind.MIXED
    )
    result = prune(doc, classification, PruneConfig())
    assert result.kept_pages == [1, 3]
    assert result.dropped_pages == [2]


def test_prune_can_disable_blank_page_dropping() -> None:
    doc = LoadedPdf(source="doc.pdf", raw_bytes=b"%PDF-fake", page_texts=["text", ""])
    classification = classify_document(doc)
    result = prune(doc, classification, PruneConfig(drop_blank_native_pages=False))
    assert result.kept_pages == [1, 2]
    assert result.dropped_pages == []


def test_prune_chunks_kept_pages(large_native_pdf_bytes: bytes) -> None:
    doc = load_pdf_bytes(large_native_pdf_bytes, source="large.pdf")
    classification = classify_document(doc)
    result = prune(doc, classification, PruneConfig(max_pages_per_chunk=15))
    assert doc.page_count == 40
    assert result.kept_pages == list(range(1, 41))
    assert [len(c) for c in result.chunks] == [15, 15, 10]
    assert result.chunks[0][0] == 1
    assert result.chunks[-1][-1] == 40


def test_extract_chunk_bytes_returns_valid_pdf_with_selected_pages(
    large_native_pdf_bytes: bytes,
) -> None:
    doc = load_pdf_bytes(large_native_pdf_bytes, source="large.pdf")
    chunk_bytes = extract_chunk_bytes(doc, [3, 4, 5])
    reader = PdfReader(io.BytesIO(chunk_bytes))
    assert len(reader.pages) == 3
    assert "Page 3" in (reader.pages[0].extract_text() or "")
