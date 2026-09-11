"""Tests for docai_poc.evaluation.metrics."""

from __future__ import annotations

import pytest

from docai_poc.evaluation.metrics import evaluate_documents, match_field
from docai_poc.schemas import (
    Entity,
    EntityFieldSpec,
    EntitySchema,
    ExtractionMethod,
    ExtractionResult,
)


def test_match_field_exact_matches_only() -> None:
    predicted = [Entity(field_name="total", raw_value="100.00")]
    truth = [Entity(field_name="total", raw_value="100.00")]
    stats = match_field(predicted, truth, "total")
    assert (stats.true_positives, stats.false_positives, stats.false_negatives) == (1, 0, 0)


def test_match_field_normalized_value_used_for_comparison() -> None:
    predicted = [Entity(field_name="total", raw_value="$100.00", normalized_value="100.00")]
    truth = [Entity(field_name="total", raw_value="100.00")]
    stats = match_field(predicted, truth, "total")
    assert stats.true_positives == 1


def test_match_field_missed_value_is_false_negative() -> None:
    predicted: list[Entity] = []
    truth = [Entity(field_name="total", raw_value="100.00")]
    stats = match_field(predicted, truth, "total")
    assert (stats.true_positives, stats.false_positives, stats.false_negatives) == (0, 0, 1)


def test_match_field_wrong_value_is_fp_and_fn() -> None:
    predicted = [Entity(field_name="total", raw_value="99.00")]
    truth = [Entity(field_name="total", raw_value="100.00")]
    stats = match_field(predicted, truth, "total")
    assert (stats.true_positives, stats.false_positives, stats.false_negatives) == (0, 1, 1)


def test_match_field_handles_duplicate_values_as_multiset() -> None:
    predicted = [
        Entity(field_name="line_item", raw_value="widget"),
        Entity(field_name="line_item", raw_value="widget"),
        Entity(field_name="line_item", raw_value="gadget"),
    ]
    truth = [
        Entity(field_name="line_item", raw_value="widget"),
        Entity(field_name="line_item", raw_value="gadget"),
        Entity(field_name="line_item", raw_value="gadget"),
    ]
    stats = match_field(predicted, truth, "line_item")
    # widget: 2 predicted / 1 truth -> 1 tp, 1 fp; gadget: 1 predicted / 2 truth -> 1 tp, 1 fn
    assert (stats.true_positives, stats.false_positives, stats.false_negatives) == (2, 1, 1)


def test_match_field_ignores_other_fields() -> None:
    predicted = [Entity(field_name="other", raw_value="x")]
    truth = [Entity(field_name="total", raw_value="100.00")]
    stats = match_field(predicted, truth, "total")
    assert (stats.true_positives, stats.false_positives, stats.false_negatives) == (0, 0, 1)


@pytest.fixture
def schema() -> EntitySchema:
    return EntitySchema(
        name="invoice_v1",
        fields=[EntityFieldSpec(name="invoice_number"), EntityFieldSpec(name="total")],
    )


def test_evaluate_documents_aggregates_across_documents(schema: EntitySchema) -> None:
    results = [
        ExtractionResult(
            document_id="doc-1",
            schema_name="invoice_v1",
            method=ExtractionMethod.NATIVE_SYNC,
            entities=[
                Entity(field_name="invoice_number", raw_value="INV-1"),
                Entity(field_name="total", raw_value="100.00"),
            ],
            latency_ms=100.0,
            page_count=1,
        ),
        ExtractionResult(
            document_id="doc-2",
            schema_name="invoice_v1",
            method=ExtractionMethod.NATIVE_SYNC,
            entities=[
                Entity(field_name="invoice_number", raw_value="INV-WRONG"),
            ],
            latency_ms=300.0,
            page_count=1,
        ),
    ]
    ground_truths = {
        "doc-1": [
            Entity(field_name="invoice_number", raw_value="INV-1"),
            Entity(field_name="total", raw_value="100.00"),
        ],
        "doc-2": [
            Entity(field_name="invoice_number", raw_value="INV-2"),
            Entity(field_name="total", raw_value="200.00"),
        ],
    }

    report = evaluate_documents(results, ground_truths, schema)

    assert report.n_documents == 2
    invoice_stats = next(f for f in report.field_stats if f.field_name == "invoice_number")
    total_stats = next(f for f in report.field_stats if f.field_name == "total")
    # doc-1 invoice_number matches (tp=1); doc-2 invoice_number wrong (fp=1, fn=1)
    assert (
        invoice_stats.true_positives,
        invoice_stats.false_positives,
        invoice_stats.false_negatives,
    ) == (
        1,
        1,
        1,
    )
    # doc-1 total matches (tp=1); doc-2 total missing entirely (fn=1)
    assert (
        total_stats.true_positives,
        total_stats.false_positives,
        total_stats.false_negatives,
    ) == (
        1,
        0,
        1,
    )
    assert report.latency_ms_p50 == pytest.approx(200.0)
    assert report.overall_precision == pytest.approx(2 / 3)
    assert report.overall_recall == pytest.approx(2 / 4)


def test_evaluate_documents_missing_ground_truth_counts_as_all_false_positive(
    schema: EntitySchema,
) -> None:
    results = [
        ExtractionResult(
            document_id="doc-unknown",
            schema_name="invoice_v1",
            method=ExtractionMethod.NATIVE_SYNC,
            entities=[Entity(field_name="invoice_number", raw_value="INV-1")],
            latency_ms=50.0,
            page_count=1,
        )
    ]
    report = evaluate_documents(results, ground_truths={}, schema=schema)
    invoice_stats = next(f for f in report.field_stats if f.field_name == "invoice_number")
    assert (invoice_stats.true_positives, invoice_stats.false_positives) == (0, 1)


def test_evaluate_documents_empty_results_returns_zeroed_report(schema: EntitySchema) -> None:
    report = evaluate_documents([], {}, schema)
    assert report.n_documents == 0
    assert report.latency_ms_p50 == 0.0
    assert report.overall_f1 == 0.0
