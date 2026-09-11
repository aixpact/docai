"""Runtime configuration for the Document AI extraction POC.

All Google Cloud resources referenced here (Document AI processors, GCS
staging buckets, the Vertex AI service account) are provisioned outside
this codebase by the Azure DevOps + Terraform pipeline — the Workbench
this runs in has no IAM permissions of its own. Settings only *name*
those pre-existing resources; authentication relies on Application
Default Credentials resolving to the service account already bound to
the Workbench.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, sourced from environment variables or `.env`.

    Attributes:
        gcp_project_id: Google Cloud project hosting the Document AI processors.
        gcp_location: Document AI processor region, e.g. ``"us"`` or ``"eu"``.
        native_processor_id: Custom extraction processor tuned for native
            (text-layer) PDFs.
        scanned_processor_id: Custom extraction processor tuned for scanned /
            image-only PDFs (OCR-based).
        sync_page_limit: Maximum pages Document AI accepts per synchronous
            (online) `process_document` call. Documents longer than this are
            either chunked for repeated sync calls or routed to batch.
        batch_input_bucket: GCS bucket (``gs://...``) used to stage input
            documents for batch (async) processing.
        batch_output_bucket: GCS bucket (``gs://...``) Document AI writes
            batch results to.
    """

    model_config = SettingsConfigDict(env_prefix="DOCAI_", env_file=".env", extra="ignore")

    gcp_project_id: str
    gcp_location: str = "us"
    native_processor_id: str
    scanned_processor_id: str
    sync_page_limit: int = 15
    batch_input_bucket: str | None = None
    batch_output_bucket: str | None = None

    def processor_path(self, processor_id: str) -> str:
        """Build the fully-qualified Document AI processor resource name.

        Args:
            processor_id: The short processor ID (not the full path).

        Returns:
            The resource name in the form
            ``projects/{project}/locations/{location}/processors/{id}``.
        """
        return (
            f"projects/{self.gcp_project_id}/locations/{self.gcp_location}"
            f"/processors/{processor_id}"
        )

    @property
    def api_endpoint(self) -> str:
        """Regional Document AI API endpoint for `gcp_location`."""
        return f"{self.gcp_location}-documentai.googleapis.com"
