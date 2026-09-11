# docai-poc

A lean POC evaluating **Google Document AI** for custom entity extraction
from PDFs — native (text-layer) and scanned, averaging ~70 pages —
orchestrated with the **Google Agent Development Kit (ADK)** and the
Document AI Python SDK, running inside a restricted **Google Workbench**
authenticated via a **Vertex AI service account** (Application Default
Credentials, no key files).

See [`PLAN.md`](PLAN.md) for the phased implementation plan and its
current status.

## Why

Evaluate whether Document AI's custom extractor processors can reliably
pull a defined set of entities out of mixed native/scanned PDFs, at what
latency and cost, and how much human review agreement/correction the
results need — before committing to it as a production pipeline.

## Workflow

```
open/read PDF -> classify (native vs. scanned) -> prune & chunk
    -> pick extraction method (native/OCR x sync/batch)
    -> call Document AI -> map to Entity/ExtractionResult
    -> evaluate (precision/recall/F1, latency) and/or human review
```

Each stage is an independently tested module under `src/docai_poc/`:

| Stage | Module | Feature covered |
|---|---|---|
| Ingest | `documents.loader` | Open/read a PDF, extract per-page text |
| Classify | `documents.classifier` | Native vs. scanned vs. mixed, per page and overall |
| Prune | `documents.pruner` | Drop boilerplate/blank pages, chunk to the sync page limit |
| Extract | `extraction.client`, `extraction.pipeline` | Native vs. OCR processor, sync (latency) vs. batch (cost) |
| Schema | `schemas` | Custom `EntitySchema` — the fields to fine-tune extraction against |
| Evaluate | `evaluation.metrics` | Precision/recall/F1 per field, latency p50/p95 |
| Review | `review.export`, `review.agreement` | CSV export for human review, inter-annotator agreement (Cohen's kappa) |
| Agent | `agent.adk_agent` | ADK agent exposing extraction as a conversational tool |

## Setup

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/) as the
dependency manager. From a Google Workbench (or any environment with
Application Default Credentials already resolving to a Vertex AI service
account bound to the target project):

```bash
uv sync              # creates .venv and installs the project + dev group from uv.lock
uv run pre-commit install
```

`uv sync` resolves against the committed `uv.lock`, so every install is
reproducible. Run any project command through `uv run <cmd>` (or activate
`.venv` as usual); add a runtime dependency with `uv add <package>` and a
dev-only one with `uv add --dev <package>`, then commit the updated
`pyproject.toml` and `uv.lock` together.

Configure via environment variables (or a `.env` file), all `DOCAI_`-prefixed:

| Variable | Required | Description |
|---|---|---|
| `DOCAI_GCP_PROJECT_ID` | yes | GCP project hosting the processors |
| `DOCAI_GCP_LOCATION` | no (default `us`) | Document AI processor region |
| `DOCAI_NATIVE_PROCESSOR_ID` | yes | Custom extractor tuned for native PDFs |
| `DOCAI_SCANNED_PROCESSOR_ID` | yes | Custom extractor tuned for scanned/OCR PDFs |
| `DOCAI_SYNC_PAGE_LIMIT` | no (default `15`) | Document AI's synchronous per-request page limit |
| `DOCAI_BATCH_INPUT_BUCKET` | only for batch mode | `gs://...` staging bucket for batch input |
| `DOCAI_BATCH_OUTPUT_BUCKET` | only for batch mode | `gs://...` bucket Document AI writes batch results to |

**These resources are provisioned externally** by an Azure DevOps +
Terraform pipeline — this Workbench has no IAM permissions of its own.
See [`terraform/README.md`](terraform/README.md).

## Running the tests

```bash
uv run pytest
uv run pre-commit run --all-files
```

All Google Cloud calls (`DocumentProcessorServiceClient`, GCS) are
injected and mocked in tests — the suite runs with no network access and
no real GCP project.

## Usage

### Define a schema and run extraction

```python
from docai_poc.config import Settings
from docai_poc.extraction.pipeline import ExtractionPipeline
from docai_poc.schemas import EntityFieldSpec, EntitySchema, FieldOccurrence

settings = Settings()  # reads DOCAI_* env vars
schema = EntitySchema(
    name="invoice_v1",
    fields=[
        EntityFieldSpec(name="invoice_number", occurrence=FieldOccurrence.REQUIRED_ONCE),
        EntityFieldSpec(name="total_amount", occurrence=FieldOccurrence.REQUIRED_ONCE),
        EntityFieldSpec(name="line_item", occurrence=FieldOccurrence.OPTIONAL_MULTIPLE),
    ],
)

pipeline = ExtractionPipeline(settings)
pdf_bytes = open("invoice.pdf", "rb").read()

# prefer_low_latency=True: sync calls (chunked if needed) — lowest latency.
# prefer_low_latency=False: batch for large/costly documents.
result = pipeline.run("invoice.pdf", pdf_bytes, schema, prefer_low_latency=True)
print(result.method, result.latency_ms, [e.field_name for e in result.entities])
```

### Evaluate against ground truth

```python
from docai_poc.evaluation.metrics import evaluate_documents

report = evaluate_documents([result], ground_truths={"invoice.pdf": [...]}, schema=schema)
print(report.overall_precision, report.overall_recall, report.latency_ms_p95)
```

### Human review round-trip

```python
from docai_poc.review.export import export_for_review, read_reviewed

export_for_review([result], "review.csv")  # open in a spreadsheet, fill in approved/corrected_value
ground_truth = read_reviewed("review.csv")  # feed straight into evaluate_documents
```

### Reviewer agreement

```python
from docai_poc.review.agreement import field_agreement

agreement = field_agreement(reviewer_a_entities, reviewer_b_entities, field_name="total_amount")
print(agreement.agreement_rate, agreement.cohens_kappa)
```

### Conversational agent (ADK)

```python
from docai_poc.agent.adk_agent import build_extraction_agent

agent = build_extraction_agent(settings, schema)  # run via the ADK runner/CLI of your choice
```

## Design principles

- **Lean POC**: each module does one thing; no speculative abstraction.
- **Typed + validated**: Pydantic models everywhere data crosses a
  boundary (config, domain schemas, results); `pyright` in `basic` mode.
- **Mockable by construction**: `DocumentAiClient` and `ExtractionPipeline`
  take injectable clients, so tests never touch real GCP.
- **Pre-commit gated**: `ruff` (lint + format) and `pyright` run on every
  commit via `.pre-commit-config.yaml`.

## Known limitations / next steps beyond this POC

- No Terraform is executed here (see [`terraform/README.md`](terraform/README.md));
  processor IDs and buckets must already exist.
- Custom processor *training* (labeling a dataset, running the trainer)
  happens in the Document AI console, not in this codebase — this repo
  assumes an already-trained processor per schema.
- Review is CSV-based, not a dedicated UI, per "lean POC style."
- Batch-mode page limits and long-running-operation retry/backoff are
  minimal; a production pipeline would want more robust polling and
  partial-failure handling for very large batch jobs.
