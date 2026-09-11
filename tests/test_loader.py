"""Tests for docai_poc.documents.loader."""

from __future__ import annotations

from docai_poc.documents.loader import load_pdf_bytes


def test_load_pdf_bytes_extracts_page_texts(native_pdf_bytes: bytes) -> None:
    doc = load_pdf_bytes(native_pdf_bytes, source="native.pdf")
    assert doc.source == "native.pdf"
    assert doc.page_count == 5
    assert len(doc.page_texts) == 5
    assert all("Page" in text for text in doc.page_texts)


def test_load_pdf_bytes_blank_pages_have_empty_text(scanned_pdf_bytes: bytes) -> None:
    doc = load_pdf_bytes(scanned_pdf_bytes, source="scanned.pdf")
    assert doc.page_count == 5
    assert all(text.strip() == "" for text in doc.page_texts)


def test_load_pdf_round_trips_raw_bytes(native_pdf_bytes: bytes) -> None:
    doc = load_pdf_bytes(native_pdf_bytes, source="native.pdf")
    assert doc.raw_bytes == native_pdf_bytes
