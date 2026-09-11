"""Tests for docai_poc.extraction.client.DocumentAiClient (Document AI mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from google.cloud import documentai

from docai_poc.config import Settings
from docai_poc.extraction.client import DocumentAiClient, blob_stem


@pytest.fixture
def settings() -> Settings:
    return Settings(
        gcp_project_id="test-project",
        gcp_location="us",
        native_processor_id="native-id",
        scanned_processor_id="scanned-id",
        batch_input_bucket="gs://input-bucket",
        batch_output_bucket="gs://output-bucket/prefix",
    )


def test_process_sync_builds_request_and_returns_document(settings: Settings) -> None:
    fake_document = documentai.Document(text="hello")
    processor_client = MagicMock()
    processor_client.process_document.return_value = MagicMock(document=fake_document)

    client = DocumentAiClient(settings, processor_client=processor_client)
    result = client.process_sync("native-id", b"%PDF-content", mime_type="application/pdf")

    assert result == fake_document
    request = processor_client.process_document.call_args.kwargs["request"]
    assert request.name == "projects/test-project/locations/us/processors/native-id"
    assert request.raw_document.content == b"%PDF-content"
    assert request.raw_document.mime_type == "application/pdf"


def test_upload_for_batch_writes_to_configured_bucket(settings: Settings) -> None:
    storage_client = MagicMock()
    blob = MagicMock()
    storage_client.bucket.return_value.blob.return_value = blob

    client = DocumentAiClient(settings, storage_client=storage_client)
    uri = client.upload_for_batch(b"%PDF-content", "docs/foo.pdf")

    storage_client.bucket.assert_called_once_with("input-bucket")
    storage_client.bucket.return_value.blob.assert_called_once_with("docs/foo.pdf")
    blob.upload_from_string.assert_called_once_with(b"%PDF-content", content_type="application/pdf")
    assert uri == "gs://input-bucket/docs/foo.pdf"


def test_upload_for_batch_without_bucket_raises() -> None:
    settings = Settings(
        gcp_project_id="p",
        native_processor_id="n",
        scanned_processor_id="s",
    )
    client = DocumentAiClient(settings, storage_client=MagicMock())
    with pytest.raises(ValueError, match="BATCH_INPUT_BUCKET"):
        client.upload_for_batch(b"x", "y.pdf")


def test_process_batch_polls_operation_and_reads_output(settings: Settings) -> None:
    fake_document = documentai.Document(text="batch result")
    fake_blob = MagicMock()
    fake_blob.name = "prefix/foo/0.json"
    fake_blob.download_as_bytes.return_value = documentai.Document.to_json(fake_document).encode(
        "utf-8"
    )

    processor_client = MagicMock()
    operation = MagicMock()
    processor_client.batch_process_documents.return_value = operation

    storage_client = MagicMock()
    storage_client.list_blobs.return_value = [fake_blob]

    client = DocumentAiClient(
        settings, processor_client=processor_client, storage_client=storage_client
    )
    results = client.process_batch("native-id", "gs://input-bucket/docs/foo.pdf")

    operation.result.assert_called_once()
    assert len(results) == 1
    assert results[0].text == "batch result"


def test_process_batch_without_output_bucket_raises() -> None:
    settings = Settings(
        gcp_project_id="p",
        native_processor_id="n",
        scanned_processor_id="s",
    )
    client = DocumentAiClient(settings, processor_client=MagicMock())
    with pytest.raises(ValueError, match="BATCH_OUTPUT_BUCKET"):
        client.process_batch("native-id", "gs://input-bucket/x.pdf")


def test_blob_stem_strips_path_and_extension() -> None:
    assert blob_stem("gs://bucket/a/b/c.pdf") == "c"
