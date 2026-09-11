# Cookbook

Runnable, self-contained examples for each use case this POC covers.
Every script uses `FakeDocumentAiClient` (see `_fake_client.py`) instead
of a real Document AI processor, so the whole cookbook runs with **zero
GCP access** — useful for onboarding, demos, or sanity-checking the API
after a change. Swap in a real, already-provisioned `DocumentAiClient`
(drop the `client=` override — `ExtractionPipeline` builds one from
`Settings` via Application Default Credentials) to run against an actual
project.

Run any example with:

```bash
uv run python cookbook/<script>.py
```

| Script | Use case |
|---|---|
| `01_basic_extraction.py` | Run the full pipeline on one PDF and print extracted entities |
| `02_latency_vs_cost_tradeoff.py` | Same large document, `prefer_low_latency=True` vs. `False` |
| `03_evaluate_against_ground_truth.py` | Score a batch of results with `evaluation.metrics.evaluate_documents` |
| `04_human_review_roundtrip.py` | Export to CSV, simulate a reviewer, read the review back as ground truth |
| `05_reviewer_agreement.py` | Cohen's kappa / agreement rate between two annotators for one field |
| `06_adk_agent_tool.py` | Call the ADK `extract_entities` tool function directly, and build the `Agent` |

`tests/test_cookbook.py` runs every script's `main()` as a smoke test, so
the cookbook stays correct as the library's API evolves.
