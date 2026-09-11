"""Cookbook: export extraction results for review, then read the review back.

`export_for_review` writes one CSV row per extracted entity with blank
`approved`/`corrected_value` columns; a reviewer fills those in (in a
spreadsheet, in practice), and `read_reviewed` turns the result back into
ground-truth `Entity` lists usable by `evaluation.metrics.evaluate_documents`.

Run: uv run python cookbook/04_human_review_roundtrip.py
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from _fake_client import FakeDocumentAiClient, make_sample_pdf

from docai_poc.config import Settings
from docai_poc.extraction.pipeline import ExtractionPipeline
from docai_poc.review.export import REVIEW_FIELDNAMES, export_for_review, read_reviewed
from docai_poc.schemas import EntityFieldSpec, EntitySchema


def _simulate_review(csv_path: Path) -> None:
    """Stand in for a human editing the CSV in a spreadsheet."""
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        if row["field_name"] == "invoice_number":
            row["approved"] = "yes"
        elif row["field_name"] == "total_amount":
            row["corrected_value"] = "1234.99"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


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
    result = pipeline.run("invoice_a.pdf", make_sample_pdf(), schema)

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "review.csv"
        export_for_review([result], csv_path)
        print("--- exported for review ---")
        print(csv_path.read_text(encoding="utf-8"))

        # Simulate a reviewer: approve the invoice_number row as-is, and
        # correct the total_amount row (a spreadsheet edit would do this).
        _simulate_review(csv_path)

        reviewed = read_reviewed(csv_path)
        print("--- reviewed ground truth ---")
        for entity in reviewed["invoice_a.pdf"]:
            print(f"  {entity.field_name}: {entity.raw_value}")


if __name__ == "__main__":
    main()
