# Infrastructure: provisioned by Azure DevOps, not from here

This Workbench has **no IAM permissions**. Every GCP resource this
codebase talks to — Document AI processors, the Vertex AI service
account, and the batch-mode GCS buckets — is created and permissioned by
a separate **Azure DevOps pipeline running Terraform**, outside this
repository and outside this Workbench's control.

This directory does **not** get applied from here. `main.tf.example` is
documentation only: it names the resources `docai_poc.config.Settings`
expects to already exist, so the ADO/Terraform pipeline owner has a
single reference for what to provision and how the resulting IDs map to
this app's environment variables.

## Resources this codebase expects to already exist

| Resource | Purpose | Consumed via |
|---|---|---|
| Document AI custom extractor processor (native/text PDFs) | Low-latency extraction for text-layer PDFs | `DOCAI_NATIVE_PROCESSOR_ID` |
| Document AI custom extractor processor (OCR/scanned PDFs) | OCR-based extraction for scanned or mixed PDFs | `DOCAI_SCANNED_PROCESSOR_ID` |
| GCS bucket — batch input staging | Source documents for async `batchProcess` calls | `DOCAI_BATCH_INPUT_BUCKET` |
| GCS bucket — batch output | Document AI writes batch results (sharded JSON) here | `DOCAI_BATCH_OUTPUT_BUCKET` |
| Vertex AI service account, bound to this Workbench | Application Default Credentials for both Document AI and GCS calls | implicit (ADC) |

The service account needs (at minimum) `roles/documentai.apiUser` on the
processors and `roles/storage.objectAdmin` scoped to the two buckets
above — granted by the Terraform pipeline, never requested or assumed
by application code.

## Why no Terraform runs here

- The Workbench's restricted IAM means `terraform apply` (or even
  `terraform plan` against real state) cannot succeed from this
  environment.
- Provisioning is centralized in Azure DevOps so processor schemas,
  bucket lifecycle policies, and IAM bindings stay under change control
  outside individual Workbenches.

If a processor ID or bucket name changes, update the corresponding
`DOCAI_*` environment variable for this app — `main.tf.example` is not a
substitute for the real ADO-managed Terraform state and should not be
copied into a working configuration as-is.
