"""Precision/recall/F1 entity matching and latency aggregation.

Ground truth is represented the same way as extraction output — a list
of `Entity` objects — so annotations produced by a human reviewer (see
`docai_poc.review`) can be evaluated with exactly the same code path as
extraction results.
"""

from __future__ import annotations

import math
from collections import Counter

from docai_poc.schemas import Entity, EntitySchema, EvalReport, ExtractionResult, FieldEvalStats


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile, matching numpy's default method."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower, upper = math.floor(rank), math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    return ordered[lower] * (upper - rank) + ordered[upper] * (rank - lower)


def match_field(
    predicted: list[Entity], ground_truth: list[Entity], field_name: str
) -> FieldEvalStats:
    """Compare predicted vs. ground-truth values for one schema field.

    Values are compared as a multiset (via `Entity.comparable_value`) so
    that duplicate values and multi-valued fields are scored correctly,
    independent of extraction order.

    Args:
        predicted: All entities extracted for a document.
        ground_truth: All annotated (true) entities for the same document.
        field_name: The schema field to score.

    Returns:
        True/false positive/negative counts for this field.
    """
    predicted_values = Counter(e.comparable_value for e in predicted if e.field_name == field_name)
    true_values = Counter(e.comparable_value for e in ground_truth if e.field_name == field_name)
    true_positives = sum((predicted_values & true_values).values())
    false_positives = sum(predicted_values.values()) - true_positives
    false_negatives = sum(true_values.values()) - true_positives
    return FieldEvalStats(
        field_name=field_name,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def _merge_stats(a: FieldEvalStats, b: FieldEvalStats) -> FieldEvalStats:
    return FieldEvalStats(
        field_name=a.field_name,
        true_positives=a.true_positives + b.true_positives,
        false_positives=a.false_positives + b.false_positives,
        false_negatives=a.false_negatives + b.false_negatives,
    )


def evaluate_documents(
    results: list[ExtractionResult],
    ground_truths: dict[str, list[Entity]],
    schema: EntitySchema,
) -> EvalReport:
    """Score a batch of extraction results against ground-truth annotations.

    Args:
        results: One `ExtractionResult` per evaluated document.
        ground_truths: Ground-truth entities keyed by
            `ExtractionResult.document_id`. A document with no key is
            treated as having no ground-truth entities (every predicted
            value counts as a false positive).
        schema: The schema being evaluated; determines which fields are
            scored.

    Returns:
        Aggregate precision/recall/F1 per field plus latency percentiles.
    """
    field_totals: dict[str, FieldEvalStats] = {
        field_name: FieldEvalStats(
            field_name=field_name, true_positives=0, false_positives=0, false_negatives=0
        )
        for field_name in schema.field_names
    }

    for result in results:
        truth = ground_truths.get(result.document_id, [])
        for field_name in schema.field_names:
            stats = match_field(result.entities, truth, field_name)
            field_totals[field_name] = _merge_stats(field_totals[field_name], stats)

    latencies = [r.latency_ms for r in results]
    return EvalReport(
        schema_name=schema.name,
        n_documents=len(results),
        field_stats=list(field_totals.values()),
        latency_ms_p50=_percentile(latencies, 0.5),
        latency_ms_p95=_percentile(latencies, 0.95),
    )
