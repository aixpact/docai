"""Thin, mockable wrapper over the Document AI and GCS Python SDKs.

Keeps every direct Google Cloud SDK call in one place so the rest of the
codebase (and its tests) can depend on this narrow interface instead of
`google.cloud.documentai`/`google.cloud.storage` directly. Authentication
uses Application Default Credentials — the Vertex AI service account
already bound to the Workbench — never explicit key files.
"""

from __future__ import annotations

from google.api_core.client_options import ClientOptions
from google.cloud import documentai, storage

from docai_poc.config import Settings

DEFAULT_MIME_TYPE = "application/pdf"


def _split_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """Split a `gs://bucket/path` URI into `(bucket, path)`."""
    without_scheme = gcs_uri.removeprefix("gs://")
    bucket, _, path = without_scheme.partition("/")
    return bucket, path


class DocumentAiClient:
    """Wraps the Document AI processor client (and GCS for batch mode).

    Both underlying clients are constructed lazily and can be injected
    (e.g. with fakes/mocks) for testing.
    """

    def __init__(
        self,
        settings: Settings,
        processor_client: documentai.DocumentProcessorServiceClient | None = None,
        storage_client: storage.Client | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            settings: Resolved application settings (project, location,
                processor IDs, batch buckets).
            processor_client: Optional pre-built Document AI client;
                built lazily against `settings.api_endpoint` otherwise.
            storage_client: Optional pre-built GCS client, used only by
                batch-mode helpers; built lazily otherwise.
        """
        self._settings = settings
        self._processor_client = processor_client
        self._storage_client = storage_client

    @property
    def processor_client(self) -> documentai.DocumentProcessorServiceClient:
        """The underlying Document AI processor client, built on first use."""
        if self._processor_client is None:
            self._processor_client = documentai.DocumentProcessorServiceClient(
                client_options=ClientOptions(api_endpoint=self._settings.api_endpoint)
            )
        return self._processor_client

    @property
    def storage_client(self) -> storage.Client:
        """The underlying GCS client, built on first use (batch mode only)."""
        if self._storage_client is None:
            self._storage_client = storage.Client(project=self._settings.gcp_project_id)
        return self._storage_client

    def process_sync(
        self, processor_id: str, content: bytes, mime_type: str = DEFAULT_MIME_TYPE
    ) -> documentai.Document:
        """Run synchronous (online) Document AI processing.

        Args:
            processor_id: Short processor ID (not the full resource path).
            content: Raw PDF bytes, already pruned/chunked to fit the
                processor's synchronous page limit.
            mime_type: Content MIME type.

        Returns:
            The extracted `documentai.Document`.
        """
        request = documentai.ProcessRequest(
            name=self._settings.processor_path(processor_id),
            raw_document=documentai.RawDocument(content=content, mime_type=mime_type),
        )
        result = self.processor_client.process_document(request=request)
        return result.document

    def upload_for_batch(self, content: bytes, blob_name: str) -> str:
        """Stage a document in the configured batch-input GCS bucket.

        Args:
            content: Raw PDF bytes to upload.
            blob_name: Destination object name within the input bucket.

        Returns:
            The `gs://` URI of the uploaded object.

        Raises:
            ValueError: If no batch input bucket is configured.
        """
        if not self._settings.batch_input_bucket:
            raise ValueError("DOCAI_BATCH_INPUT_BUCKET is not configured")
        bucket_name, _ = _split_gcs_uri(self._settings.batch_input_bucket)
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(content, content_type=DEFAULT_MIME_TYPE)
        return f"gs://{bucket_name}/{blob_name}"

    def process_batch(
        self,
        processor_id: str,
        gcs_input_uri: str,
        mime_type: str = DEFAULT_MIME_TYPE,
        operation_timeout_s: float = 600.0,
    ) -> list[documentai.Document]:
        """Run asynchronous (batch) Document AI processing.

        Cost-efficient for large or high-volume documents that would
        otherwise need many chunked synchronous calls; trades latency
        (this call blocks on the long-running operation) for lower
        per-page overhead.

        Args:
            processor_id: Short processor ID (not the full resource path).
            gcs_input_uri: `gs://` URI of the document to process (see
                `upload_for_batch`).
            mime_type: Content MIME type.
            operation_timeout_s: Max seconds to wait for the batch
                operation to complete.

        Returns:
            One `documentai.Document` per output shard Document AI wrote.

        Raises:
            ValueError: If no batch output bucket is configured.
        """
        if not self._settings.batch_output_bucket:
            raise ValueError("DOCAI_BATCH_OUTPUT_BUCKET is not configured")
        output_bucket, output_prefix = _split_gcs_uri(self._settings.batch_output_bucket)
        output_uri_prefix = (
            f"gs://{output_bucket}/{output_prefix.rstrip('/')}/{blob_stem(gcs_input_uri)}/"
        )

        request = documentai.BatchProcessRequest(
            name=self._settings.processor_path(processor_id),
            input_documents=documentai.BatchDocumentsInputConfig(
                gcs_documents=documentai.GcsDocuments(
                    documents=[documentai.GcsDocument(gcs_uri=gcs_input_uri, mime_type=mime_type)]
                )
            ),
            document_output_config=documentai.DocumentOutputConfig(
                gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(
                    gcs_uri=output_uri_prefix
                )
            ),
        )
        operation = self.processor_client.batch_process_documents(request=request)
        operation.result(timeout=operation_timeout_s)
        return self._fetch_batch_output(output_bucket, output_prefix=output_uri_prefix)

    def _fetch_batch_output(
        self, bucket_name: str, output_prefix: str
    ) -> list[documentai.Document]:
        _, prefix = _split_gcs_uri(output_prefix)
        blobs = self.storage_client.list_blobs(bucket_name, prefix=prefix)
        documents = []
        for blob in blobs:
            if blob.name.endswith(".json"):
                documents.append(
                    documentai.Document.from_json(
                        blob.download_as_bytes(), ignore_unknown_fields=True
                    )
                )
        return documents


def blob_stem(gcs_uri: str) -> str:
    """Filename (without extension) of a `gs://` object, for output prefixing."""
    _, path = _split_gcs_uri(gcs_uri)
    name = path.rsplit("/", maxsplit=1)[-1]
    return name.rsplit(".", maxsplit=1)[0]
