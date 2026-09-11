"""Tests for docai_poc.schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from docai_poc.schemas import (
    Entity,
    EntityFieldSpec,
    EntitySchema,
    EvalReport,
    ExtractionMethod,
    ExtractionResult,
    FieldEvalStats,
    FieldOccurrence,
)


def test_entity_schema_field_names_and_required() -> None:
    schema = EntitySchema(
        name="invoice_v1",
        fields=[
            EntityFieldSpec(name="invoice_number", occurrence=FieldOccurrence.REQUIRED_ONCE),
            EntityFieldSpec(name="line_item", occurrence=FieldOccurrence.OPTIONAL_MULTIPLE),
        ],
    )
    assert schema.field_names == ["invoice_number", "line_item"]
    assert schema.required_field_names == ["invoice_number"]
    assert schema.fields[1].is_multi_valued is True
    assert schema.fields[0].is_multi_valued is False


def test_entity_schema_rejects_duplicate_field_names() -> None:
    with pytest.raises(ValidationError, match="duplicate field names"):
        EntitySchema(
            name="dup",
            fields=[
                EntityFieldSpec(name="a"),
                EntityFieldSpec(name="a"),
            ],
        )


def test_entity_schema_requires_at_least_one_field() -> None:
    with pytest.raises(ValidationError):
        EntitySchema(name="empty", fields=[])


def test_extraction_method_flags() -> None:
    assert ExtractionMethod.NATIVE_SYNC.is_native is True
    assert ExtractionMethod.NATIVE_SYNC.is_sync is True
    assert ExtractionMethod.OCR_BATCH.is_native is False
    assert ExtractionMethod.OCR_BATCH.is_sync is False


def test_entity_comparable_value_prefers_normalized() -> None:
    entity = Entity(field_name="total", raw_value="$1,200.00", normalized_value="1200.00")
    assert entity.comparable_value == "1200.00"

    entity_no_norm = Entity(field_name="total", raw_value="  Acme Corp ")
    assert entity_no_norm.comparable_value == "acme corp"


def test_entity_confidence_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        Entity(field_name="total", raw_value="x", confidence=1.5)


def test_extraction_result_values_for() -> None:
    result = ExtractionResult(
        document_id="doc-1",
        schema_name="invoice_v1",
        method=ExtractionMethod.NATIVE_SYNC,
        entities=[
            Entity(field_name="line_item", raw_value="widget"),
            Entity(field_name="line_item", raw_value="gadget"),
            Entity(field_name="invoice_number", raw_value="INV-1"),
        ],
        latency_ms=123.4,
        page_count=3,
    )
    assert [e.raw_value for e in result.values_for("line_item")] == ["widget", "gadget"]
    assert result.values_for("missing_field") == []


def test_field_eval_stats_precision_recall_f1() -> None:
    stats = FieldEvalStats(
        field_name="invoice_number", true_positives=8, false_positives=2, false_negatives=2
    )
    assert stats.precision == pytest.approx(0.8)
    assert stats.recall == pytest.approx(0.8)
    assert stats.f1 == pytest.approx(0.8)


def test_field_eval_stats_handles_zero_denominators() -> None:
    stats = FieldEvalStats(field_name="x", true_positives=0, false_positives=0, false_negatives=0)
    assert stats.precision == 0.0
    assert stats.recall == 0.0
    assert stats.f1 == 0.0


def test_eval_report_micro_averages() -> None:
    report = EvalReport(
        schema_name="invoice_v1",
        n_documents=2,
        field_stats=[
            FieldEvalStats(field_name="a", true_positives=9, false_positives=1, false_negatives=1),
            FieldEvalStats(field_name="b", true_positives=1, false_positives=1, false_negatives=1),
        ],
        latency_ms_p50=100.0,
        latency_ms_p95=200.0,
    )
    # tp=10, fp=2, fn=2 -> precision=recall=10/12
    assert report.overall_precision == pytest.approx(10 / 12)
    assert report.overall_recall == pytest.approx(10 / 12)
    assert report.overall_f1 == pytest.approx(10 / 12)
