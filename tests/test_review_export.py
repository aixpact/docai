"""Tests for docai_poc.review.export."""

from __future__ import annotations

from pathlib import Path

from docai_poc.review.export import export_for_review, read_reviewed
from docai_poc.schemas import Entity, ExtractionMethod, ExtractionResult


def _result(document_id: str) -> ExtractionResult:
    return ExtractionResult(
        document_id=document_id,
        schema_name="invoice_v1",
        method=ExtractionMethod.NATIVE_SYNC,
        entities=[
            Entity(field_name="invoice_number", raw_value="INV-1", confidence=0.9, page_number=1),
            Entity(field_name="total", raw_value="100.00", confidence=0.8),
        ],
        latency_ms=100.0,
        page_count=1,
    )


def test_export_for_review_writes_one_row_per_entity(tmp_path: Path) -> None:
    csv_path = tmp_path / "review.csv"
    export_for_review([_result("doc-1")], csv_path)

    content = csv_path.read_text(encoding="utf-8")
    rows = content.strip().splitlines()
    assert rows[0].split(",") == [
        "document_id",
        "field_name",
        "raw_value",
        "normalized_value",
        "confidence",
        "page_number",
        "approved",
        "corrected_value",
    ]
    assert len(rows) == 3  # header + 2 entities


def test_read_reviewed_skips_unreviewed_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "review.csv"
    export_for_review([_result("doc-1")], csv_path)

    reviewed = read_reviewed(csv_path)
    assert reviewed == {}


def test_read_reviewed_includes_approved_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "review.csv"
    export_for_review([_result("doc-1")], csv_path)
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    assert header[-2:] == ["approved", "corrected_value"]
    # Mark the invoice_number row (first data row) approved.
    lines[1] = "doc-1,invoice_number,INV-1,,0.9,1,yes,"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    reviewed = read_reviewed(csv_path)
    assert "doc-1" in reviewed
    assert len(reviewed["doc-1"]) == 1
    entity = reviewed["doc-1"][0]
    assert entity.field_name == "invoice_number"
    assert entity.raw_value == "INV-1"
    assert entity.confidence == 1.0
    assert entity.page_number == 1


def test_read_reviewed_uses_corrected_value_when_present(tmp_path: Path) -> None:
    csv_path = tmp_path / "review.csv"
    export_for_review([_result("doc-1")], csv_path)
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    lines[2] = "doc-1,total,100.00,,0.8,,,150.00"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    reviewed = read_reviewed(csv_path)
    entity = reviewed["doc-1"][0]
    assert entity.raw_value == "150.00"
    assert entity.page_number is None
