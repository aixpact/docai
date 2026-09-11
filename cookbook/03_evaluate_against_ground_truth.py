"""Cookbook: score extraction results against ground-truth annotations.

`evaluate_documents` compares a batch of `ExtractionResult`s against
ground-truth entities keyed by document ID, producing per-field
precision/recall/F1 plus latency percentiles across the batch.

Run: uv run python cookbook/03_evaluate_against_ground_truth.py
"""

from __future__ import annotations

from _fake_client import FakeDocumentAiClient, make_sample_pdf

from docai_poc.config import Settings
from docai_poc.evaluation.metrics import evaluate_documents
from docai_poc.extraction.pipeline import ExtractionPipeline
from docai_poc.schemas import Entity, EntityFieldSpec, EntitySchema


def main() -> None:
    settings = Settings(
        gcp_project_id="demo-project",
        native_processor_id="native-processor-demo",
        scanned_processor_id="scanned-processor-demo",
    )
    schema = EntitySchema(
        name="invoice_v1",
        fields=[EntityFieldSpec(name="invoice_number"), EntityFieldSpec(name="total_amount")],
    )
    pipeline = ExtractionPipeline(settings, client=FakeDocumentAiClient())

    results = [
        pipeline.run("invoice_a.pdf", make_sample_pdf(label="Invoice A"), schema),
        pipeline.run("invoice_b.pdf", make_sample_pdf(label="Invoice B"), schema),
    ]

    # In practice this comes from `docai_poc.review.export.read_reviewed`
    # (see 04_human_review_roundtrip.py) or a hand-labeled test set.
    ground_truths = {
        "invoice_a.pdf": [
            Entity(field_name="invoice_number", raw_value="INV-2024-0001"),
            Entity(field_name="total_amount", raw_value="1234.56"),
        ],
        "invoice_b.pdf": [
            # The fake processor always returns the same canned value, so
            # this document's "true" invoice number deliberately differs
            # to show a miss show up as a false positive + false negative.
            Entity(field_name="invoice_number", raw_value="INV-2024-9999"),
            Entity(field_name="total_amount", raw_value="1234.56"),
        ],
    }

    report = evaluate_documents(results, ground_truths, schema)

    print(f"documents evaluated: {report.n_documents}")
    print(
        f"overall precision/recall/F1: {report.overall_precision:.2f} / "
        f"{report.overall_recall:.2f} / {report.overall_f1:.2f}"
    )
    print(f"latency p50/p95 (ms): {report.latency_ms_p50:.2f} / {report.latency_ms_p95:.2f}")
    for field_stats in report.field_stats:
        print(
            f"  {field_stats.field_name}: precision={field_stats.precision:.2f} "
            f"recall={field_stats.recall:.2f} f1={field_stats.f1:.2f}"
        )


if __name__ == "__main__":
    main()
