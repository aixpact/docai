"""Cookbook: run the extraction pipeline end to end on one PDF.

Run: uv run python cookbook/01_basic_extraction.py
"""

from __future__ import annotations

from _fake_client import FakeDocumentAiClient, make_sample_pdf

from docai_poc.config import Settings
from docai_poc.extraction.pipeline import ExtractionPipeline
from docai_poc.schemas import EntityFieldSpec, EntitySchema, FieldOccurrence


def main() -> None:
    settings = Settings(
        gcp_project_id="demo-project",
        native_processor_id="native-processor-demo",
        scanned_processor_id="scanned-processor-demo",
    )
    schema = EntitySchema(
        name="invoice_v1",
        fields=[
            EntityFieldSpec(name="invoice_number", occurrence=FieldOccurrence.REQUIRED_ONCE),
            EntityFieldSpec(name="total_amount", occurrence=FieldOccurrence.REQUIRED_ONCE),
        ],
    )

    # Swap FakeDocumentAiClient() for no `client=` argument at all to hit a
    # real, already-provisioned Document AI processor instead.
    pipeline = ExtractionPipeline(settings, client=FakeDocumentAiClient())
    pdf_bytes = make_sample_pdf(num_pages=3)

    result = pipeline.run("sample_invoice.pdf", pdf_bytes, schema)

    print(f"method: {result.method.value}")
    print(f"page_count: {result.page_count}, pruned_page_count: {result.pruned_page_count}")
    print(f"latency_ms: {result.latency_ms:.2f}")
    for entity in result.entities:
        print(f"  {entity.field_name}: {entity.raw_value} (confidence={entity.confidence})")


if __name__ == "__main__":
    main()
