"""Orchestrates the end-to-end extraction workflow.

Workflow: open/read the document, classify + prune it, pick an
extraction method (native vs. OCR processor; sync vs. batch), call
Document AI, and map the result onto our own `Entity`/`ExtractionResult`
models.
"""

from __future__ import annotations

import time
import uuid

from google.cloud import documentai

from docai_poc.config import Settings
from docai_poc.documents.classifier import DocumentClassification, DocumentKind, classify_document
from docai_poc.documents.loader import LoadedPdf, load_pdf_bytes
from docai_poc.documents.pruner import PruneConfig, PruneResult, extract_chunk_bytes, prune
from docai_poc.extraction.client import DocumentAiClient
from docai_poc.schemas import Entity, EntitySchema, ExtractionMethod, ExtractionResult


def choose_extraction_method(
    classification: DocumentClassification,
    page_count: int,
    sync_page_limit: int,
    prefer_low_latency: bool = True,
) -> ExtractionMethod:
    """Pick native-vs-OCR and sync-vs-batch for a classified document.

    Native-only processing is only used for documents classified fully
    `NATIVE`; `MIXED` and `SCANNED` documents both route to the OCR-capable
    processor, since a native-only processor can miss content on scanned
    pages. Sync is preferred whenever it fits Document AI's per-request
    page limit or `prefer_low_latency` is set (in which case oversized
    documents are still processed via repeated, chunked sync calls);
    otherwise a single batch call is used, trading latency (async,
    poll-until-done) for lower per-call overhead on large documents.

    Args:
        classification: Output of `classify_document` for this document.
        page_count: Total page count of the document.
        sync_page_limit: Document AI's synchronous per-request page limit.
        prefer_low_latency: If `True`, favor (possibly chunked) sync calls
            over a single batch call.

    Returns:
        The `ExtractionMethod` to use.
    """
    is_native = classification.kind == DocumentKind.NATIVE
    fits_sync = page_count <= sync_page_limit

    if fits_sync or prefer_low_latency:
        return ExtractionMethod.NATIVE_SYNC if is_native else ExtractionMethod.OCR_SYNC
    return ExtractionMethod.NATIVE_BATCH if is_native else ExtractionMethod.OCR_BATCH


def _map_entities(document: documentai.Document, page_offset: int = 0) -> list[Entity]:
    entities = []
    for e in document.entities:
        page_number = None
        if e.page_anchor and e.page_anchor.page_refs:
            page_number = int(e.page_anchor.page_refs[0].page) + 1 + page_offset
        normalized = (
            e.normalized_value.text if e.normalized_value and e.normalized_value.text else None
        )
        entities.append(
            Entity(
                field_name=e.type_,
                raw_value=e.mention_text,
                normalized_value=normalized,
                confidence=e.confidence,
                page_number=page_number,
            )
        )
    return entities


class ExtractionPipeline:
    """Runs the load -> classify -> prune -> extract workflow."""

    def __init__(self, settings: Settings, client: DocumentAiClient | None = None) -> None:
        """Initialize the pipeline.

        Args:
            settings: Resolved application settings.
            client: Optional pre-built `DocumentAiClient` (inject a fake
                for testing); built from `settings` otherwise.
        """
        self._settings = settings
        self._client = client or DocumentAiClient(settings)

    def run(
        self,
        document_id: str,
        pdf_bytes: bytes,
        schema: EntitySchema,
        prefer_low_latency: bool = True,
        prune_config: PruneConfig | None = None,
    ) -> ExtractionResult:
        """Run the full extraction workflow on one document.

        Args:
            document_id: Caller-supplied identifier (e.g. filename).
            pdf_bytes: Raw PDF content.
            schema: The custom entity schema being evaluated/used.
            prefer_low_latency: Passed through to `choose_extraction_method`.
            prune_config: Optional pruning configuration; defaults to
                `PruneConfig()`.

        Returns:
            The extraction result, including latency and page counts.
        """
        start = time.perf_counter()

        loaded = load_pdf_bytes(pdf_bytes, source=document_id)
        classification = classify_document(loaded)
        prune_result = prune(loaded, classification, prune_config)
        method = choose_extraction_method(
            classification, loaded.page_count, self._settings.sync_page_limit, prefer_low_latency
        )
        processor_id = (
            self._settings.native_processor_id
            if method.is_native
            else self._settings.scanned_processor_id
        )

        if method.is_sync:
            entities = self._extract_via_sync_chunks(loaded, prune_result, processor_id)
        else:
            entities = self._extract_via_batch(loaded, processor_id)

        latency_ms = (time.perf_counter() - start) * 1000
        return ExtractionResult(
            document_id=document_id,
            schema_name=schema.name,
            method=method,
            entities=entities,
            latency_ms=latency_ms,
            page_count=loaded.page_count,
            pruned_page_count=len(prune_result.kept_pages),
        )

    def _extract_via_sync_chunks(
        self, loaded: LoadedPdf, prune_result: PruneResult, processor_id: str
    ) -> list[Entity]:
        entities: list[Entity] = []
        for chunk in prune_result.chunks:
            chunk_bytes = extract_chunk_bytes(loaded, chunk)
            document = self._client.process_sync(processor_id, chunk_bytes)
            page_offset = chunk[0] - 1
            entities.extend(_map_entities(document, page_offset=page_offset))
        return entities

    def _extract_via_batch(self, loaded: LoadedPdf, processor_id: str) -> list[Entity]:
        blob_name = f"docai-poc/{uuid.uuid4()}.pdf"
        gcs_uri = self._client.upload_for_batch(loaded.raw_bytes, blob_name)
        documents = self._client.process_batch(processor_id, gcs_uri)
        entities: list[Entity] = []
        for document in documents:
            entities.extend(_map_entities(document))
        return entities
