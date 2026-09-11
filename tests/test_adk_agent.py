"""Tests for docai_poc.agent.adk_agent.

Only the tool function's own behavior is tested here (pipeline mocked);
this deliberately does not exercise the ADK LLM loop itself.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from docai_poc.agent.adk_agent import build_extract_entities_tool, build_extraction_agent
from docai_poc.config import Settings
from docai_poc.schemas import EntityFieldSpec, EntitySchema, ExtractionMethod, ExtractionResult


def _schema() -> EntitySchema:
    return EntitySchema(name="invoice_v1", fields=[EntityFieldSpec(name="invoice_number")])


def test_extract_entities_tool_reads_file_and_calls_pipeline(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-fake-content")

    schema = _schema()
    fake_result = ExtractionResult(
        document_id=str(pdf_path),
        schema_name=schema.name,
        method=ExtractionMethod.NATIVE_SYNC,
        entities=[],
        latency_ms=42.0,
        page_count=1,
    )
    pipeline = MagicMock()
    pipeline.run.return_value = fake_result

    tool = build_extract_entities_tool(pipeline, schema)
    output = tool(str(pdf_path), True)

    pipeline.run.assert_called_once_with(
        document_id=str(pdf_path),
        pdf_bytes=b"%PDF-fake-content",
        schema=schema,
        prefer_low_latency=True,
    )
    assert output["document_id"] == str(pdf_path)
    assert output["method"] == "native_sync"
    assert output["latency_ms"] == 42.0


def test_extract_entities_tool_defaults_prefer_low_latency_true(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF")
    schema = _schema()
    pipeline = MagicMock()
    pipeline.run.return_value = ExtractionResult(
        document_id=str(pdf_path),
        schema_name=schema.name,
        method=ExtractionMethod.NATIVE_SYNC,
        entities=[],
        latency_ms=1.0,
        page_count=1,
    )

    tool = build_extract_entities_tool(pipeline, schema)
    tool(str(pdf_path))

    assert pipeline.run.call_args.kwargs["prefer_low_latency"] is True


def test_build_extraction_agent_wires_name_model_and_tool() -> None:
    settings = Settings(gcp_project_id="p", native_processor_id="n", scanned_processor_id="s")
    schema = _schema()

    agent = build_extraction_agent(settings, schema, model="gemini-2.0-flash")

    assert agent.name == "docai_extraction_agent"
    assert agent.model == "gemini-2.0-flash"
    assert len(agent.tools) == 1
    assert isinstance(agent.instruction, str)
    assert schema.name in agent.instruction
