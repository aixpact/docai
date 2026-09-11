"""Tests for docai_poc.extraction.pipeline (Document AI client mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from google.cloud import documentai

from docai_poc.config import Settings
from docai_poc.documents.classifier import DocumentClassification, DocumentKind
from docai_poc.documents.pruner import PruneConfig
from docai_poc.extraction.pipeline import ExtractionPipeline, choose_extraction_method
from docai_poc.schemas import EntityFieldSpec, EntitySchema, ExtractionMethod


def _classification(kind: DocumentKind, n_pages: int) -> DocumentClassification:
    is_native = kind == DocumentKind.NATIVE
    return DocumentClassification(page_is_native=[is_native] * n_pages, kind=kind)


@pytest.mark.parametrize(
    ("kind", "page_count", "sync_limit", "prefer_low_latency", "expected"),
    [
        (DocumentKind.NATIVE, 5, 15, True, ExtractionMethod.NATIVE_SYNC),
        (DocumentKind.NATIVE, 5, 15, False, ExtractionMethod.NATIVE_SYNC),
        (DocumentKind.NATIVE, 40, 15, True, ExtractionMethod.NATIVE_SYNC),
        (DocumentKind.NATIVE, 40, 15, False, ExtractionMethod.NATIVE_BATCH),
        (DocumentKind.SCANNED, 5, 15, True, ExtractionMethod.OCR_SYNC),
        (DocumentKind.SCANNED, 40, 15, False, ExtractionMethod.OCR_BATCH),
        (DocumentKind.MIXED, 40, 15, False, ExtractionMethod.OCR_BATCH),
        (DocumentKind.MIXED, 5, 15, False, ExtractionMethod.OCR_SYNC),
    ],
)
def test_choose_extraction_method(
    kind: DocumentKind,
    page_count: int,
    sync_limit: int,
    prefer_low_latency: bool,
    expected: ExtractionMethod,
) -> None:
    classification = _classification(kind, page_count)
    assert (
        choose_extraction_method(classification, page_count, sync_limit, prefer_low_latency)
        == expected
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        gcp_project_id="p", native_processor_id="native-id", scanned_processor_id="scanned-id"
    )


@pytest.fixture
def schema() -> EntitySchema:
    return EntitySchema(name="invoice_v1", fields=[EntityFieldSpec(name="invoice_number")])


def _entity_document(field_name: str, value: str, page: int = 0) -> documentai.Document:
    return documentai.Document(
        entities=[
            documentai.Document.Entity(
                type_=field_name,
                mention_text=value,
                confidence=0.95,
                page_anchor=documentai.Document.PageAnchor(
                    page_refs=[documentai.Document.PageAnchor.PageRef(page=page)]
                ),
            )
        ]
    )


def test_pipeline_run_native_sync_single_chunk(
    settings: Settings, schema: EntitySchema, native_pdf_bytes: bytes
) -> None:
    fake_client = MagicMock()
    fake_client.process_sync.return_value = _entity_document("invoice_number", "INV-1", page=1)

    pipeline = ExtractionPipeline(settings, client=fake_client)
    result = pipeline.run("doc.pdf", native_pdf_bytes, schema)

    assert result.method == ExtractionMethod.NATIVE_SYNC
    assert result.page_count == 5
    assert result.pruned_page_count == 5
    assert fake_client.process_sync.call_count == 1
    assert fake_client.process_sync.call_args.args[0] == "native-id"
    assert len(result.entities) == 1
    entity = result.entities[0]
    assert entity.field_name == "invoice_number"
    assert entity.raw_value == "INV-1"
    assert entity.page_number == 2  # page 1 (0-indexed) within the single chunk starting at page 1


def test_pipeline_run_chunks_large_native_document(
    settings: Settings, schema: EntitySchema, large_native_pdf_bytes: bytes
) -> None:
    fake_client = MagicMock()
    fake_client.process_sync.side_effect = [
        _entity_document("invoice_number", "chunk-a", page=0),
        _entity_document("invoice_number", "chunk-b", page=0),
        _entity_document("invoice_number", "chunk-c", page=0),
    ]

    pipeline = ExtractionPipeline(settings, client=fake_client)
    result = pipeline.run("large.pdf", large_native_pdf_bytes, schema)

    assert result.method == ExtractionMethod.NATIVE_SYNC
    assert fake_client.process_sync.call_count == 3
    # Page offsets: chunks start at pages 1, 16, 31 -> entity page 0 maps to 1, 16, 31.
    pages = sorted(e.page_number for e in result.entities if e.page_number is not None)
    assert pages == [1, 16, 31]


def test_pipeline_run_scanned_document_uses_ocr_processor(
    settings: Settings, schema: EntitySchema, scanned_pdf_bytes: bytes
) -> None:
    fake_client = MagicMock()
    fake_client.process_sync.return_value = documentai.Document()

    pipeline = ExtractionPipeline(settings, client=fake_client)
    result = pipeline.run("scanned.pdf", scanned_pdf_bytes, schema)

    assert result.method == ExtractionMethod.OCR_SYNC
    assert fake_client.process_sync.call_args.args[0] == "scanned-id"
    assert result.entities == []


def test_pipeline_run_batch_when_cost_preferred_for_large_doc(
    settings: Settings, schema: EntitySchema, large_native_pdf_bytes: bytes
) -> None:
    fake_client = MagicMock()
    fake_client.upload_for_batch.return_value = "gs://input-bucket/docs/large.pdf"
    fake_client.process_batch.return_value = [_entity_document("invoice_number", "batch-value")]

    pipeline = ExtractionPipeline(settings, client=fake_client)
    result = pipeline.run("large.pdf", large_native_pdf_bytes, schema, prefer_low_latency=False)

    assert result.method == ExtractionMethod.NATIVE_BATCH
    fake_client.upload_for_batch.assert_called_once()
    fake_client.process_batch.assert_called_once_with(
        "native-id", "gs://input-bucket/docs/large.pdf"
    )
    assert fake_client.process_sync.call_count == 0
    assert len(result.entities) == 1
    assert result.entities[0].raw_value == "batch-value"


def test_pipeline_applies_prune_config(
    settings: Settings, schema: EntitySchema, native_pdf_bytes: bytes
) -> None:
    fake_client = MagicMock()
    fake_client.process_sync.return_value = documentai.Document()

    pipeline = ExtractionPipeline(settings, client=fake_client)
    result = pipeline.run(
        "doc.pdf",
        native_pdf_bytes,
        schema,
        prune_config=PruneConfig(drop_pages=frozenset({1, 2})),
    )

    assert result.page_count == 5
    assert result.pruned_page_count == 3
