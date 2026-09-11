"""Tests for docai_poc.review.agreement."""

from __future__ import annotations

import pytest

from docai_poc.review.agreement import cohens_kappa, field_agreement
from docai_poc.schemas import Entity


def test_cohens_kappa_perfect_agreement() -> None:
    assert cohens_kappa(["a", "b", "a", "c"], ["a", "b", "a", "c"]) == pytest.approx(1.0)


def test_cohens_kappa_no_items_returns_none() -> None:
    assert cohens_kappa([], []) is None


def test_cohens_kappa_chance_level_agreement_is_near_zero() -> None:
    # Two independent-looking coders over a balanced binary label.
    a = ["yes", "no", "yes", "no", "yes", "no", "yes", "no"]
    b = ["yes", "yes", "no", "no", "no", "no", "yes", "yes"]
    kappa = cohens_kappa(a, b)
    assert kappa is not None
    assert -0.5 < kappa < 0.5


def test_cohens_kappa_all_same_label_and_agree_is_one() -> None:
    assert cohens_kappa(["x", "x", "x"], ["x", "x", "x"]) == 1.0


def test_cohens_kappa_below_chance_agreement_is_negative() -> None:
    assert cohens_kappa(["x", "y"], ["y", "x"]) == pytest.approx(-1.0)


def _entities(
    doc_values: dict[str, str | None], field_name: str = "total"
) -> dict[str, list[Entity]]:
    result: dict[str, list[Entity]] = {}
    for doc_id, value in doc_values.items():
        result[doc_id] = (
            [Entity(field_name=field_name, raw_value=value)] if value is not None else []
        )
    return result


def test_field_agreement_only_compares_shared_documents() -> None:
    a = _entities({"doc-1": "100.00", "doc-2": "50.00", "doc-3": "10.00"})
    b = _entities({"doc-1": "100.00", "doc-2": "999.00"})  # doc-3 not reviewed by b
    result = field_agreement(a, b, "total")
    assert result.n_items == 2
    assert result.n_agreed == 1
    assert result.agreement_rate == pytest.approx(0.5)


def test_field_agreement_missing_field_counts_as_disagreement() -> None:
    a = _entities({"doc-1": "100.00"})
    b = _entities({"doc-1": None})  # annotator b found nothing for this field
    result = field_agreement(a, b, "total")
    assert result.n_items == 1
    assert result.n_agreed == 0


def test_field_agreement_no_shared_documents() -> None:
    a = _entities({"doc-1": "100.00"})
    b = _entities({"doc-2": "100.00"})
    result = field_agreement(a, b, "total")
    assert result.n_items == 0
    assert result.agreement_rate == 0.0
    assert result.cohens_kappa is None
