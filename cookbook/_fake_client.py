"""Shared fake Document AI client for the cookbook examples.

Real Document AI processors and GCS buckets are provisioned externally
(see `terraform/README.md`) and this cookbook has no access to them, so
every example swaps in `FakeDocumentAiClient` — a canned in-memory stand-in
matching `docai_poc.extraction.client.DocumentAiClient`'s interface — to
exercise the exact same pipeline code a production run would use. Point
`ExtractionPipeline` at a real `DocumentAiClient` (drop the `client=`
override) to run against an actual project.
"""

from __future__ import annotations

import io

from google.cloud import documentai
from reportlab.pdfgen import canvas

# Keyed by processor_id, so different processors can return different
# canned results in an example that exercises native vs. scanned routing.
CANNED_ENTITIES: dict[str, list[tuple[str, str, float]]] = {
    "native-processor-demo": [
        ("invoice_number", "INV-2024-0001", 0.97),
        ("total_amount", "1234.56", 0.94),
    ],
    "scanned-processor-demo": [
        ("invoice_number", "INV-2024-0002", 0.81),
    ],
}


class FakeDocumentAiClient:
    """Stands in for `DocumentAiClient` in the cookbook examples."""

    def __init__(self) -> None:
        self.sync_calls: list[str] = []
        self.batch_calls: list[str] = []

    def process_sync(
        self, processor_id: str, content: bytes, mime_type: str = "application/pdf"
    ) -> documentai.Document:
        self.sync_calls.append(processor_id)
        return _canned_document(processor_id)

    def upload_for_batch(self, content: bytes, blob_name: str) -> str:
        return f"gs://fake-bucket/{blob_name}"

    def process_batch(
        self, processor_id: str, gcs_input_uri: str, mime_type: str = "application/pdf"
    ) -> list[documentai.Document]:
        self.batch_calls.append(processor_id)
        return [_canned_document(processor_id)]


def _canned_document(processor_id: str) -> documentai.Document:
    entities = [
        documentai.Document.Entity(type_=name, mention_text=value, confidence=confidence)
        for name, value, confidence in CANNED_ENTITIES.get(processor_id, [])
    ]
    return documentai.Document(entities=entities)


def make_sample_pdf(num_pages: int = 3, label: str = "Sample Invoice") -> bytes:
    """Build a small in-memory PDF with a real text layer, for the examples.

    Draws enough text per page to clear `classify_document`'s native-page
    character threshold, so these examples reliably route through the
    native (not scanned/OCR) processor.
    """
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(612, 792))
    for page in range(num_pages):
        pdf.drawString(72, 720, f"{label} - page {page + 1} of {num_pages}")
        pdf.drawString(72, 700, "Billed to: Example Customer, 123 Main Street, Springfield")
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()
