"""Tests for docai_poc.config.Settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from docai_poc.config import Settings


def _env(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    defaults = {
        "DOCAI_GCP_PROJECT_ID": "test-project",
        "DOCAI_NATIVE_PROCESSOR_ID": "native-proc-id",
        "DOCAI_SCANNED_PROCESSOR_ID": "scanned-proc-id",
    }
    defaults.update(overrides)
    for key, value in defaults.items():
        monkeypatch.setenv(key, value)


def test_settings_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch)
    settings = Settings()  # type: ignore[call-arg]
    assert settings.gcp_project_id == "test-project"
    assert settings.gcp_location == "us"
    assert settings.sync_page_limit == 15
    assert settings.batch_input_bucket is None


def test_settings_missing_required_field_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOCAI_GCP_PROJECT_ID", raising=False)
    monkeypatch.delenv("DOCAI_NATIVE_PROCESSOR_ID", raising=False)
    monkeypatch.delenv("DOCAI_SCANNED_PROCESSOR_ID", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_processor_path_builds_resource_name(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, DOCAI_GCP_LOCATION="eu")
    settings = Settings()  # type: ignore[call-arg]
    assert settings.processor_path("abc123") == (
        "projects/test-project/locations/eu/processors/abc123"
    )


def test_api_endpoint_reflects_location(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, DOCAI_GCP_LOCATION="eu")
    settings = Settings()  # type: ignore[call-arg]
    assert settings.api_endpoint == "eu-documentai.googleapis.com"
