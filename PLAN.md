# Document AI Entity Extraction POC — Implementation Plan

Lean POC evaluating Google Document AI for custom entity extraction from
~70-page PDFs (mixed native-text and scanned/image content), orchestrated
with the Google Agent Development Kit (ADK) and the Python Document AI SDK,
running inside a restricted Google Workbench (Vertex AI service-account
auth, no IAM; all GCP resources provisioned externally by Azure DevOps +
Terraform).

Checklist convention: `[x]` done, `[ ]` pending. Each top-level step is
built on its own local feature branch, covered by pytest (mocked GCP
calls), linted/type-checked via pre-commit (ruff + pyright), then merged
into `claude/documentai-entity-extraction-wexhem` (this repo's designated
integration branch, used here in place of a separate `develop`).

## 0. Project scaffolding — `feat/scaffolding`
- [x] `pyproject.toml` (deps, ruff/pyright config, pytest config)
- [x] `.pre-commit-config.yaml` (ruff lint+format, pyright)
- [x] Package skeleton under `src/docai_poc/`
- [x] `tests/` skeleton + `conftest.py` fixtures

## 1. Configuration & auth — `feat/config`
- [x] `config.py`: `pydantic-settings` `Settings` for project id, location,
      processor IDs (native vs OCR/scanned, keyed by schema), sync page
      limit, batch GCS staging bucket. Reads from env vars / `.env`;
      credentials come from Application Default Credentials (Vertex AI
      service account already bound to the Workbench — no key files).
- [x] Unit tests for settings loading/validation.

## 2. Domain schemas — `feat/schemas`
- [x] `schemas.py`: Pydantic models — `EntityFieldSpec`, `EntitySchema`
      (custom schema definition used to fine-tune what a processor should
      extract and to drive evaluation), `Entity`, `ExtractionMethod` enum
      (native sync / native batch / ocr sync / ocr batch), `ExtractionResult`,
      `EvalReport`.
- [x] Unit tests (validation edge cases).

## 3. Document ingestion — `feat/documents`
- [x] `documents/loader.py`: open/read a PDF from local path or bytes,
      page count, per-page text extraction (via `pypdf`).
- [x] `documents/classifier.py`: classify each page / the document as
      `native` vs `scanned` using a text-density heuristic (chars/page
      threshold) — drives which processor + method is used.
- [x] `documents/pruner.py`: prune/chunk the document — drop
      caller-specified boilerplate pages (e.g. cover/TOC/blank), then split
      into page-range chunks that respect Document AI's synchronous
      page-count limit, so a 70-page doc can still use low-latency sync
      calls chunk-by-chunk, or be routed whole to batch.
- [x] Unit tests using small in-memory PDFs generated with `reportlab`.

## 4. Extraction — `feat/extraction`
- [x] `extraction/client.py`: thin wrapper interface over
      `google.cloud.documentai.DocumentProcessorServiceClient` —
      `process_sync()` (online, low latency) and `process_batch()`
      (async, GCS in/out, cost-efficient for large/scanned docs) — chooses
      processor id from `Settings` based on native/scanned + user's
      latency-vs-cost preference.
- [x] `extraction/pipeline.py`: orchestrates
      load → classify → prune → pick method → call client → map the
      returned `Document` proto entities onto our `Entity`/`ExtractionResult`
      models, recording wall-clock latency.
- [x] Unit tests with a fully mocked Document AI client (no network/GCP).

## 5. Evaluation — `feat/evaluation`
- [x] `evaluation/metrics.py`: match extracted entities against a
      ground-truth `EntitySchema`-shaped annotation set (exact + normalized
      string match per field), compute precision/recall/F1 per field and
      overall, plus latency aggregation (p50/p95) across a batch of runs.
- [x] Unit tests covering exact match, miss, and mismatch cases.

## 6. Human review & agreement — `feat/review`
- [x] `review/export.py`: export an `ExtractionResult` (or a batch of them)
      to CSV for human reviewers to approve/correct field values.
- [x] `review/agreement.py`: inter-annotator agreement between two
      reviewers (or reviewer vs model) per field — exact-match agreement
      rate and Cohen's kappa for categorical fields.
- [x] Unit tests.

## 7. ADK agent wrapper — `feat/agent`
- [x] `agent/adk_agent.py`: minimal `google-adk` `Agent` exposing the
      extraction pipeline as a callable tool (`extract_entities`), so the
      POC can be driven conversationally from within the Workbench.
- [x] Unit test that the tool function itself (not the LLM loop) behaves
      correctly, with the pipeline mocked.

## 8. Infra note (no execution here) — `docs/terraform-note`
- [x] `terraform/README.md` explaining that all Document AI
      processors/buckets/service-account bindings are provisioned by the
      Azure DevOps + Terraform pipeline (this Workbench has no IAM), plus
      an illustrative (non-applied) `main.tf.example` naming the resources
      this codebase expects to already exist.

## 9. Docs & wrap-up — `docs/readme`
- [x] `README.md`: goal, architecture diagram (text), workflow, feature
      matrix, setup (Workbench + ADC), running tests, evaluation/review
      usage, known limitations / next steps beyond POC scope.
- [x] Final `pre-commit run --all-files` + `pytest` clean on the
      integration branch; push `claude/documentai-entity-extraction-wexhem`.

## Explicitly out of scope for this POC
- Provisioning/Terraform execution (blocked by Workbench IAM restrictions —
  owned by the Azure DevOps pipeline).
- Custom processor schema *training* UI — we model the schema and call an
  already-trained processor; fine-tuning the trained model itself happens
  in the Document AI console/dataset, outside this codebase.
- A production review UI (a CSV-based human-in-the-loop flow stands in for
  it, per "lean POC style").
