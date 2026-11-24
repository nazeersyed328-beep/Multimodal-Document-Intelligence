# Portfolio Project Summary

## Short summary

Multimodal Document Intelligence is a local-first Streamlit application that
turns PDFs and images into schema-validated structured records. It combines
careful document validation and preprocessing, Gemini vision extraction through
the official SDK, strict Pydantic schemas, deterministic quality rules,
session-only human review, and a labeled evaluation harness.

## What it demonstrates

- Multimodal ingestion for PDFs, PNG, JPEG, and WEBP with signature/MIME checks.
- PDF rendering and image normalization before any model call.
- Provider-neutral extraction design with an SDK 2.x Gemini adapter.
- Safe JSON parsing, bounded correction retries, and strict typed outputs.
- Allowlisted, sanitized provider diagnostics instead of raw exception exposure.
- Deterministic quality assessment that is explicitly separate from model accuracy.
- Human review that preserves the original extraction in every export.
- Local, mocked, and opt-in live evaluation with mode labels and manually authored ground truth.
- Hermetic test coverage using AppTest and mocked provider responses.

## Architecture in one paragraph

`app.py` owns session state and user actions. `src/document` validates and
prepares bytes; `src/extraction` sends prepared page images through a
provider-neutral interface and validates returned JSON; `src/quality` applies
deterministic checks and review overlays; `src/evaluation` reuses the pipeline
with fixture or live providers and writes labeled reports. Configuration,
schemas, and logging are isolated under `src/config.py`, `src/schemas/`, and
`src/utils/`.

## Evidence

- Final local regression: `234 passed, 1 skipped`.
- Compilation and dependency checks pass.
- Dataset validation: 11 synthetic samples, valid with no errors or warnings.
- Streamlit health endpoint: `200 ok`.
- Export verification: 19 generated JSON files, all parseable, with no secret
  patterns or forbidden key names.
- Live Gemini validation is pending: five controlled smoke attempts returned
  HTTP `503 service_unavailable`; no usable response, no live Pydantic
  validation, and no live accuracy metric exist.

## Demo narrative

Use only synthetic/local artifacts:

1. Open the app and upload a clean synthetic invoice or receipt.
2. Process and preview the page to show local validation and normalization.
3. Start a mocked extraction in tests or show a preserved local extraction result.
4. Point out strict Pydantic fields, confidence estimates, and review flags.
5. Open quality assessment and show that the rule-based indicator is not accuracy.
6. Mark an issue reviewed and edit a low-confidence field; download all four exports.
7. Validate the dataset and run a local synthetic evaluation.
8. End with the honest status: local pipeline verified, live Gemini validation pending.

Do not run a live extraction during a portfolio demo unless a successful,
currently authorized live response has been obtained and the key is protected.

## Resume bullet drafts

- Built a multimodal document-intelligence pipeline for PDFs and images using
  Streamlit, Pydantic v2, Pillow, PyMuPDF, and the official Gemini SDK.
- Implemented schema-constrained extraction with safe JSON parsing, bounded
  retries, sanitized provider diagnostics, and strict typed output models.
- Designed deterministic quality assessment and session-only human review that
  preserve original model output in labeled JSON and Markdown exports.
- Created a labeled evaluation harness with synthetic fixtures, mocked API
  orchestration, opt-in live mode, metrics, and JSON/Markdown/CSV reports.
- Verified 234 local tests, dataset validation, Streamlit health, export
  safety, and secret scanning; documented live validation as pending after
  repeated provider `503` responses.

## Scope boundaries

Do not claim:

- validated real Gemini extraction accuracy;
- production deployment or hosted service;
- authentication, database persistence, or CI;
- calibrated confidence probabilities;
- OCR, since the system relies on vision-model extraction;
- database-backed review history, since review is session-only.

Accurate framing:

> A locally complete multimodal extraction workflow with a real Gemini
> integration implemented and live validation pending because of provider
> capacity failures.

## Future work

1. Obtain one successful live Gemini response and confirm parsing/Pydantic validation.
2. Run the planned four-document live evaluation, then decide on full evaluation.
3. Add repository packaging, a license, and minimal CI.
4. Consider persistence, authentication, and deployment only as separate approved phases.
