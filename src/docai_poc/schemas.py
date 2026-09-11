"""Typed domain models shared across ingestion, extraction and evaluation.

These models are the POC's own representation — independent of the
Document AI wire format — so that pipeline, evaluation and review code
never depend directly on `google.cloud.documentai` protos.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class FieldOccurrence(str, Enum):
    """How many times a field is expected to occur in a document.

    Mirrors Document AI custom-extractor schema occurrence semantics.
    """

    OPTIONAL_ONCE = "optional_once"
    REQUIRED_ONCE = "required_once"
    OPTIONAL_MULTIPLE = "optional_multiple"
    REQUIRED_MULTIPLE = "required_multiple"


class EntityFieldSpec(BaseModel):
    """A single field in a custom extraction schema.

    Attributes:
        name: Machine-readable field name (matches the processor's trained
            entity type).
        description: Human-readable description, used when fine-tuning the
            schema and by reviewers.
        occurrence: Expected cardinality of this field within a document.
    """

    name: str = Field(min_length=1)
    description: str = ""
    occurrence: FieldOccurrence = FieldOccurrence.OPTIONAL_ONCE

    @property
    def is_multi_valued(self) -> bool:
        """Whether this field may legitimately appear more than once."""
        return self.occurrence in (
            FieldOccurrence.OPTIONAL_MULTIPLE,
            FieldOccurrence.REQUIRED_MULTIPLE,
        )


class EntitySchema(BaseModel):
    """A named, versioned custom entity schema to extract and evaluate.

    Attributes:
        name: Schema identifier, e.g. ``"invoice_v1"``.
        version: Free-form schema version string.
        fields: The fields this schema defines. Field names must be unique.
    """

    name: str = Field(min_length=1)
    version: str = "1.0.0"
    fields: list[EntityFieldSpec] = Field(min_length=1)

    @field_validator("fields")
    @classmethod
    def _unique_field_names(cls, fields: list[EntityFieldSpec]) -> list[EntityFieldSpec]:
        names = [f.name for f in fields]
        if len(names) != len(set(names)):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate field names in schema: {duplicates}")
        return fields

    @property
    def field_names(self) -> list[str]:
        """Names of all fields defined by this schema."""
        return [f.name for f in self.fields]

    @property
    def required_field_names(self) -> list[str]:
        """Names of fields whose occurrence marks them as required."""
        return [
            f.name
            for f in self.fields
            if f.occurrence in (FieldOccurrence.REQUIRED_ONCE, FieldOccurrence.REQUIRED_MULTIPLE)
        ]


class ExtractionMethod(str, Enum):
    """Which Document AI processing path produced a result.

    Native vs. OCR selects the processor tuned for text-layer vs.
    scanned/image content; sync vs. batch trades latency against cost and
    document-size limits.
    """

    NATIVE_SYNC = "native_sync"
    NATIVE_BATCH = "native_batch"
    OCR_SYNC = "ocr_sync"
    OCR_BATCH = "ocr_batch"

    @property
    def is_native(self) -> bool:
        """Whether this method targets the native (text-layer) processor."""
        return self in (ExtractionMethod.NATIVE_SYNC, ExtractionMethod.NATIVE_BATCH)

    @property
    def is_sync(self) -> bool:
        """Whether this method calls the synchronous (online) API."""
        return self in (ExtractionMethod.NATIVE_SYNC, ExtractionMethod.OCR_SYNC)


class Entity(BaseModel):
    """A single extracted (or annotated) entity value.

    Attributes:
        field_name: The `EntityFieldSpec.name` this value belongs to.
        raw_value: The text value as extracted/annotated.
        normalized_value: A normalized form (e.g. ISO date), when available.
        confidence: Model confidence in ``[0, 1]``; ``1.0`` for human
            annotations.
        page_number: 1-indexed page the value was found on, if known.
    """

    field_name: str = Field(min_length=1)
    raw_value: str
    normalized_value: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    page_number: int | None = Field(default=None, ge=1)

    @property
    def comparable_value(self) -> str:
        """The value to use for matching: normalized form if present."""
        return (self.normalized_value or self.raw_value).strip().lower()


class ExtractionResult(BaseModel):
    """The outcome of running the extraction pipeline on one document.

    Attributes:
        document_id: Caller-supplied identifier (e.g. filename).
        schema_name: Name of the `EntitySchema` used.
        method: The extraction method actually used.
        entities: Extracted entities.
        latency_ms: Wall-clock time for the Document AI call(s), in
            milliseconds.
        page_count: Total pages in the source document.
        pruned_page_count: Pages actually sent to Document AI after
            pruning, if pruning was applied.
    """

    document_id: str
    schema_name: str
    method: ExtractionMethod
    entities: list[Entity] = Field(default_factory=list)
    latency_ms: float = Field(ge=0.0)
    page_count: int = Field(ge=0)
    pruned_page_count: int | None = Field(default=None, ge=0)

    def values_for(self, field_name: str) -> list[Entity]:
        """All extracted entities for a given field name."""
        return [e for e in self.entities if e.field_name == field_name]


class FieldEvalStats(BaseModel):
    """Precision/recall/F1 for a single schema field."""

    field_name: str
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)

    @property
    def precision(self) -> float:
        """Fraction of extracted values for this field that were correct."""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        """Fraction of ground-truth values for this field that were found."""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


class EvalReport(BaseModel):
    """Aggregate evaluation results across one or more documents.

    Attributes:
        schema_name: Name of the `EntitySchema` evaluated against.
        n_documents: Number of documents included in this report.
        field_stats: Per-field precision/recall/F1 breakdown.
        latency_ms_p50: Median extraction latency across documents.
        latency_ms_p95: 95th-percentile extraction latency across documents.
    """

    schema_name: str
    n_documents: int = Field(ge=0)
    field_stats: list[FieldEvalStats] = Field(default_factory=list)
    latency_ms_p50: float = Field(ge=0.0)
    latency_ms_p95: float = Field(ge=0.0)

    @property
    def overall_precision(self) -> float:
        """Micro-averaged precision across all fields."""
        tp = sum(f.true_positives for f in self.field_stats)
        fp = sum(f.false_positives for f in self.field_stats)
        return tp / (tp + fp) if (tp + fp) else 0.0

    @property
    def overall_recall(self) -> float:
        """Micro-averaged recall across all fields."""
        tp = sum(f.true_positives for f in self.field_stats)
        fn = sum(f.false_negatives for f in self.field_stats)
        return tp / (tp + fn) if (tp + fn) else 0.0

    @property
    def overall_f1(self) -> float:
        """Micro-averaged F1 across all fields."""
        p, r = self.overall_precision, self.overall_recall
        return 2 * p * r / (p + r) if (p + r) else 0.0
