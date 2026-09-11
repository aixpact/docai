"""Cookbook: exposing extraction as an ADK tool.

Builds the same `extract_entities` tool function `build_extraction_agent`
wires into an ADK `Agent`, and calls it directly the way ADK's function
tool dispatcher would — without running the LLM loop, so this example
needs no model credentials. See `docai_poc.agent.adk_agent` for how to
wire the resulting `Agent` into an ADK `Runner` in a real Workbench.

Run: uv run python cookbook/06_adk_agent_tool.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from _fake_client import FakeDocumentAiClient, make_sample_pdf

from docai_poc.agent.adk_agent import build_extract_entities_tool, build_extraction_agent
from docai_poc.config import Settings
from docai_poc.extraction.pipeline import ExtractionPipeline
from docai_poc.schemas import EntityFieldSpec, EntitySchema


def main() -> None:
    settings = Settings(
        gcp_project_id="demo-project",
        native_processor_id="native-processor-demo",
        scanned_processor_id="scanned-processor-demo",
    )
    schema = EntitySchema(name="invoice_v1", fields=[EntityFieldSpec(name="invoice_number")])

    # The tool takes a plain pipeline, so a fake client works here exactly
    # as it does for a direct pipeline.run() call.
    pipeline = ExtractionPipeline(settings, client=FakeDocumentAiClient())
    extract_entities = build_extract_entities_tool(pipeline, schema)

    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = Path(tmp_dir) / "invoice.pdf"
        pdf_path.write_bytes(make_sample_pdf())

        output = extract_entities(str(pdf_path), True)
        print("tool output:")
        for key in ("document_id", "method", "latency_ms"):
            print(f"  {key}: {output[key]}")
        for entity in output["entities"]:
            print(f"  entity: {entity['field_name']} = {entity['raw_value']}")

    # Wiring the same tool into a real conversational agent (needs a model
    # ID and, to actually run, ADK's Runner + credentials — not exercised
    # here):
    agent = build_extraction_agent(settings, schema)
    print(f"\nbuilt agent {agent.name!r} with model {agent.model!r} and {len(agent.tools)} tool(s)")


if __name__ == "__main__":
    main()
