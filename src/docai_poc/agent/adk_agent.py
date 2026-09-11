"""ADK agent exposing the extraction pipeline as a callable tool.

Lets a Workbench user drive the POC conversationally (e.g. "extract the
invoice fields from this PDF") while all Document AI logic stays in
`docai_poc.extraction.pipeline`; the agent only translates
natural-language requests into calls to a single `extract_entities` tool.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from google.adk.agents import Agent

from docai_poc.config import Settings
from docai_poc.extraction.pipeline import ExtractionPipeline
from docai_poc.schemas import EntitySchema

DEFAULT_MODEL = "gemini-2.0-flash"


def build_extract_entities_tool(
    pipeline: ExtractionPipeline, schema: EntitySchema
) -> Callable[..., dict[str, Any]]:
    """Build an `extract_entities` tool function bound to a pipeline and schema.

    ADK inspects a tool function's signature and docstring to describe it
    to the model, so the returned function takes only JSON-serializable
    arguments and returns a plain dict.

    Args:
        pipeline: The extraction pipeline to run.
        schema: The custom entity schema to extract and report against.

    Returns:
        A function suitable for use in `Agent(tools=[...])`.
    """

    def extract_entities(pdf_path: str, prefer_low_latency: bool = True) -> dict[str, Any]:
        """Extract this agent's schema entities from a PDF document.

        Args:
            pdf_path: Path to the PDF file to process.
            prefer_low_latency: If true, prefer faster (possibly chunked
                synchronous) processing; if false, prefer the more
                cost-efficient batch method for large documents.

        Returns:
            A dict with the document id, extraction method used, latency
            in milliseconds, and the list of extracted entities (each
            with field_name, raw_value, normalized_value, confidence and
            page_number).
        """
        pdf_bytes = Path(pdf_path).read_bytes()
        result = pipeline.run(
            document_id=pdf_path,
            pdf_bytes=pdf_bytes,
            schema=schema,
            prefer_low_latency=prefer_low_latency,
        )
        return result.model_dump(mode="json")

    return extract_entities


def build_extraction_agent(
    settings: Settings, schema: EntitySchema, model: str = DEFAULT_MODEL
) -> Agent:
    """Build a minimal ADK agent that exposes extraction as a tool.

    Args:
        settings: Resolved application settings.
        schema: The custom entity schema this agent extracts.
        model: Model ID backing the agent's LLM (served via Vertex AI).

    Returns:
        A ready-to-run ADK `Agent`.
    """
    pipeline = ExtractionPipeline(settings)
    tool = build_extract_entities_tool(pipeline, schema)
    return Agent(
        name="docai_extraction_agent",
        model=model,
        instruction=(
            f"You extract '{schema.name}' entities from PDF documents using the "
            "extract_entities tool. Always call the tool rather than guessing "
            "values yourself."
        ),
        tools=[tool],
    )
