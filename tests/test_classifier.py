"""Tests for docai_poc.documents.classifier."""

from __future__ import annotations

from docai_poc.documents.classifier import DocumentKind, classify_document
from docai_poc.documents.loader import LoadedPdf, load_pdf_bytes


def test_classify_all_native_pages(native_pdf_bytes: bytes) -> None:
    doc = load_pdf_bytes(native_pdf_bytes, source="native.pdf")
    result = classify_document(doc)
    assert result.kind == DocumentKind.NATIVE
    assert all(result.page_is_native)
    assert result.native_page_ratio == 1.0


def test_classify_all_scanned_pages(scanned_pdf_bytes: bytes) -> None:
    doc = load_pdf_bytes(scanned_pdf_bytes, source="scanned.pdf")
    result = classify_document(doc)
    assert result.kind == DocumentKind.SCANNED
    assert not any(result.page_is_native)
    assert result.native_page_ratio == 0.0


def test_classify_mixed_document() -> None:
    doc = LoadedPdf(
        source="mixed.pdf",
        raw_bytes=b"%PDF-fake",
        page_texts=["substantial text content here", "", "more real text on this page"],
    )
    result = classify_document(doc)
    assert result.kind == DocumentKind.MIXED
    assert result.page_is_native == [True, False, True]
    assert result.native_page_ratio == 2 / 3


def test_classify_empty_document_is_scanned() -> None:
    doc = LoadedPdf(source="empty.pdf", raw_bytes=b"%PDF-fake", page_texts=[])
    result = classify_document(doc)
    assert result.kind == DocumentKind.SCANNED
    assert result.native_page_ratio == 0.0


def test_classify_respects_min_chars_threshold() -> None:
    doc = LoadedPdf(source="short.pdf", raw_bytes=b"%PDF-fake", page_texts=["hi"])
    assert classify_document(doc, min_chars_per_page=1).kind == DocumentKind.NATIVE
    assert classify_document(doc, min_chars_per_page=10).kind == DocumentKind.SCANNED
