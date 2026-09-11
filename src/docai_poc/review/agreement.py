"""Inter-annotator agreement between two reviewers (or a reviewer vs. the model).

Both annotators are represented the same way as extraction/ground-truth
output — `dict[document_id, list[Entity]]` — so this works equally for
two humans, or a human reviewer vs. the model's own extraction, which is
how model quality on a per-field basis gets tracked over time.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, Field

from docai_poc.schemas import Entity


class FieldAgreement(BaseModel):
    """Agreement between two annotators for a single schema field.

    Attributes:
        field_name: The schema field scored.
        n_items: Number of documents both annotators labeled.
        n_agreed: Number of those documents where both gave the same value.
        cohens_kappa: Chance-corrected agreement in `[-1, 1]`; `None` when
            there were no shared documents to compare.
    """

    field_name: str
    n_items: int = Field(ge=0)
    n_agreed: int = Field(ge=0)
    cohens_kappa: float | None = None

    @property
    def agreement_rate(self) -> float:
        """Fraction of shared documents where both annotators agreed."""
        return self.n_agreed / self.n_items if self.n_items else 0.0


def cohens_kappa(labels_a: Sequence[str | None], labels_b: Sequence[str | None]) -> float | None:
    """Compute Cohen's kappa for two equal-length label sequences.

    Args:
        labels_a: Labels assigned by the first annotator, one per item.
        labels_b: Labels assigned by the second annotator, aligned by index.

    Returns:
        Kappa in `[-1, 1]`, or `None` if there are no items to compare.
    """
    n = len(labels_a)
    if n == 0:
        return None

    observed_agreement = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b) / n
    categories = set(labels_a) | set(labels_b)
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    chance_agreement = sum((counts_a[c] / n) * (counts_b[c] / n) for c in categories)

    if chance_agreement >= 1.0:
        # Only possible when both annotators use a single, shared category
        # for every item, in which case observed_agreement is also 1.0.
        return 1.0
    return (observed_agreement - chance_agreement) / (1 - chance_agreement)


def _field_label(entities: list[Entity], field_name: str) -> str | None:
    values = [e.comparable_value for e in entities if e.field_name == field_name]
    return values[0] if values else None


def field_agreement(
    annotator_a: dict[str, list[Entity]],
    annotator_b: dict[str, list[Entity]],
    field_name: str,
) -> FieldAgreement:
    """Score agreement between two annotators for one schema field.

    Only documents present in both annotators' output are compared. A
    field missing from one annotator's output for a shared document
    counts as a distinct (`None`) label, so it registers as disagreement
    rather than being silently skipped.

    Args:
        annotator_a: First annotator's entities, keyed by document ID.
        annotator_b: Second annotator's entities, keyed by document ID.
        field_name: The schema field to score.

    Returns:
        Agreement rate and Cohen's kappa for this field.
    """
    document_ids = sorted(set(annotator_a) & set(annotator_b))
    labels_a = [_field_label(annotator_a[doc_id], field_name) for doc_id in document_ids]
    labels_b = [_field_label(annotator_b[doc_id], field_name) for doc_id in document_ids]

    n_items = len(document_ids)
    n_agreed = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b)
    kappa = cohens_kappa(labels_a, labels_b)
    return FieldAgreement(
        field_name=field_name, n_items=n_items, n_agreed=n_agreed, cohens_kappa=kappa
    )
