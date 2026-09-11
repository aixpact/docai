"""Cookbook: measure inter-annotator agreement between two reviewers.

`field_agreement` works on the same `dict[document_id, list[Entity]]`
shape produced by `review.export.read_reviewed`, so it applies equally
to two human reviewers or a human reviewer vs. the model's own output
(a good way to track per-field model quality over time).

Run: uv run python cookbook/05_reviewer_agreement.py
"""

from __future__ import annotations

from docai_poc.review.agreement import field_agreement
from docai_poc.schemas import Entity


def main() -> None:
    reviewer_a = {
        "invoice_a.pdf": [Entity(field_name="total_amount", raw_value="1234.56")],
        "invoice_b.pdf": [Entity(field_name="total_amount", raw_value="500.00")],
        "invoice_c.pdf": [Entity(field_name="total_amount", raw_value="75.00")],
    }
    reviewer_b = {
        "invoice_a.pdf": [Entity(field_name="total_amount", raw_value="1234.56")],  # agrees
        "invoice_b.pdf": [Entity(field_name="total_amount", raw_value="550.00")],  # disagrees
        "invoice_c.pdf": [Entity(field_name="total_amount", raw_value="75.00")],  # agrees
    }

    agreement = field_agreement(reviewer_a, reviewer_b, "total_amount")
    assert agreement.cohens_kappa is not None  # guaranteed: n_items > 0 here

    print(f"documents compared: {agreement.n_items}")
    print(f"agreement rate: {agreement.agreement_rate:.2f}")
    print(f"cohen's kappa: {agreement.cohens_kappa:.2f}")
    print(
        "-> low agreement on a field is a signal to tighten the schema "
        "description or add more training examples for that field."
    )


if __name__ == "__main__":
    main()
