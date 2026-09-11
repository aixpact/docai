"""Cookbook: choosing sync (latency) vs. batch (cost) for a large document.

`ExtractionPipeline.run(..., prefer_low_latency=...)` decides between
(possibly chunked) synchronous calls and a single batch call once a
document exceeds Document AI's synchronous page limit. This example runs
the same 40-page document both ways and shows the resulting method and
how many Document AI calls each made.

Run: uv run python cookbook/02_latency_vs_cost_tradeoff.py
"""

from __future__ import annotations

from _fake_client import FakeDocumentAiClient, make_sample_pdf

from docai_poc.config import Settings
from docai_poc.extraction.pipeline import ExtractionPipeline
from docai_poc.schemas import EntityFieldSpec, EntitySchema


def main() -> None:
    settings = Settings(
        gcp_project_id="demo-project",
        native_processor_id="native-processor-demo",
        scanned_processor_id="scanned-processor-demo",
        sync_page_limit=15,
    )
    schema = EntitySchema(name="invoice_v1", fields=[EntityFieldSpec(name="invoice_number")])
    pdf_bytes = make_sample_pdf(num_pages=40)

    for label, prefer_low_latency in (("low-latency", True), ("cost-efficient", False)):
        client = FakeDocumentAiClient()
        pipeline = ExtractionPipeline(settings, client=client)
        result = pipeline.run(
            "large_document.pdf", pdf_bytes, schema, prefer_low_latency=prefer_low_latency
        )
        print(f"[{label}] method={result.method.value}")
        print(
            f"[{label}] sync calls={len(client.sync_calls)}, batch calls={len(client.batch_calls)}"
        )


if __name__ == "__main__":
    main()
