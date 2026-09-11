"""Export extraction results to CSV for human review, and read reviews back.

Stands in for a dedicated review UI in this "lean POC" build: a reviewer
opens the CSV in a spreadsheet, fills in `approved` and/or
`corrected_value` per row, and `read_reviewed` turns that back into
ground-truth `Entity` lists usable by `docai_poc.evaluation`.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from docai_poc.schemas import Entity, ExtractionResult

REVIEW_FIELDNAMES = [
    "document_id",
    "field_name",
    "raw_value",
    "normalized_value",
    "confidence",
    "page_number",
    "approved",
    "corrected_value",
]

_TRUTHY = {"y", "yes", "true", "1"}


def export_for_review(results: list[ExtractionResult], path: str | Path) -> None:
    """Write extracted entities to a CSV for human review.

    Args:
        results: Extraction results to export, one row per entity.
        path: Destination CSV path.
    """
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDNAMES)
        writer.writeheader()
        for result in results:
            for entity in result.entities:
                writer.writerow(
                    {
                        "document_id": result.document_id,
                        "field_name": entity.field_name,
                        "raw_value": entity.raw_value,
                        "normalized_value": entity.normalized_value or "",
                        "confidence": entity.confidence,
                        "page_number": entity.page_number if entity.page_number is not None else "",
                        "approved": "",
                        "corrected_value": "",
                    }
                )


def read_reviewed(path: str | Path) -> dict[str, list[Entity]]:
    """Read a reviewed CSV back into ground-truth entities per document.

    A row becomes a ground-truth entity if the reviewer marked it
    `approved` (a truthy value) or supplied a `corrected_value`; the
    entity's value is the correction when present, otherwise the
    original extracted value. Unreviewed rows (neither approved nor
    corrected) are skipped.

    Args:
        path: Path to a CSV previously written by `export_for_review` and
            then annotated by a reviewer.

    Returns:
        Ground-truth entities keyed by `document_id`.
    """
    reviewed: dict[str, list[Entity]] = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            approved = row["approved"].strip().lower() in _TRUTHY
            corrected = row["corrected_value"].strip()
            if not approved and not corrected:
                continue
            page_number = row["page_number"].strip()
            reviewed[row["document_id"]].append(
                Entity(
                    field_name=row["field_name"],
                    raw_value=corrected or row["raw_value"],
                    normalized_value=row["normalized_value"].strip() or None,
                    confidence=1.0,
                    page_number=int(page_number) if page_number else None,
                )
            )
    return dict(reviewed)
