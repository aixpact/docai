"""Shared pytest fixtures for the docai_poc test suite."""

from __future__ import annotations

import io

import pytest
from pypdf import PdfWriter
from reportlab.pdfgen import canvas


def _text_page_pdf_bytes(num_pages: int, words_per_page: int = 400) -> bytes:
    """Build a PDF with real embedded text on every page (a "native" PDF)."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(612, 792))
    for page in range(num_pages):
        pdf.drawString(72, 720, f"Page {page + 1}")
        line = " ".join(["lorem"] * words_per_page)
        pdf.drawString(72, 700, line[:2000])
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def _blank_page_pdf_bytes(num_pages: int) -> bytes:
    """Build a PDF with no text layer at all (stands in for a scanned page)."""
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.fixture
def native_pdf_bytes() -> bytes:
    """A 5-page PDF with a real text layer on every page."""
    return _text_page_pdf_bytes(num_pages=5)


@pytest.fixture
def scanned_pdf_bytes() -> bytes:
    """A 5-page PDF with no text layer (simulating scanned images)."""
    return _blank_page_pdf_bytes(num_pages=5)


@pytest.fixture
def large_native_pdf_bytes() -> bytes:
    """A 40-page native PDF, large enough to require pruning/chunking."""
    return _text_page_pdf_bytes(num_pages=40)
