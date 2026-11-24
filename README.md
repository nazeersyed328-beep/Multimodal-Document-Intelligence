# Multimodal Document Intelligence

## 1. Overview

A local-first document-intelligence workflow that ingests PDFs and images,
prepares normalized page images, sends those images to Gemini only when
explicitly requested, validates model output with strict Pydantic schemas,
runs deterministic quality checks, supports session-only human review, and
produces labeled evaluation reports.

- Version: `0.6.0-phase6`
- Interface: Streamlit
- Runtime: Python on Windows
- Core libraries: Pydantic v2, Pillow, PyMuPDF, `google-genai`, pytest
- Design goal: honest, inspectable pipeline behavior rather than a single opaque accuracy number

## 2. Current status

**Locally ready, live validation pending.** This is not a production-ready
system and must not be presented as having validated real Gemini extraction
accuracy.

- Final local regression baseline after Phase 6 changes: `234 passed, 1 skipped`.
- Compilation, dependency consistency, dataset validation, and Streamlit health check passed.
- Five controlled live smoke attempts against `gemini-3.8-flash` returned HTTP
  `503 service_unavailable` with the same sanitized high-demand message.
- No usable live Gemini response was received, so live response parsing, live
  Pydantic validation, the four-document live evaluation, and full live
  evaluation remain incomplete.
- Synthetic and mocked results are useful for plumbing and reporting checks;
  they do not measure real model accuracy.
- There is no Git repository, CI workflow, authentication, database, or
  deployment in this workspace. The MIT `LICENSE` file is present, but its
  copyright holder placeholder needs confirmation before publication.

## 3. Implemented vs planned

| Capability | Status |
| --- | --- |
| Strict Pydantic schemas for invoices, receipts, notes, and diagrams | Implemented and locally tested |
| Environment-driven configuration with placeholder-key handling | Implemented |
| Upload validation, inspection, SHA-256 dedupe, bounded session state | Implemented |
| PDF inspection/rendering and image normalization | Implemented |
| Gemini SDK 2.x provider with safe diagnostics and bounded retries | Implemented locally; live response pending |
| Auto-detect and explicit document-type extraction | Implemented |
| Safe JSON parsing and schema validation with correction attempts | Implemented and mocked/local-tested |
| Deterministic quality assessment and rule-based indicator | Implemented |
| Session-only issue review, field edits, notes, and reset | Implemented |
| Original, quality, reviewed-output, and Markdown exports | Implemented |
| Synthetic dataset with manually authored ground truth | Implemented |
| Local, mocked, and opt-in live evaluation runners | Implemented; live run blocked |
| JSON, Markdown, and CSV evaluation reports | Implemented |
| Public repository packaging, CI, database, auth, deployment | Not implemented |
| MIT license file | Added; copyright holder placeholder needs confirmation |

## 4. Quick start

From the project root on Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
Copy-Item .env.example .env
py -m streamlit run app.py
```

If script activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Set at minimum in `.env`:

```dotenv
GEMINI_API_KEY=your-api-key-here
GEMINI_MODEL=gemini-3.8-flash
```

Never commit `.env`. Upload, preprocessing, preview, local evaluation, and
mocked evaluation work without a real key. Extraction and live evaluation
require a configured key and explicit user action.

## 5. User workflow

1. Upload one or more PDF or image documents.
2. Review accepted/rejected files and the document list.
3. Process the selected document or all valid documents.
4. Review preprocessing status and page previews.
5. In **6. AI extraction**, choose auto-detect or an explicit document type.
6. Click **Extract with Gemini** only when ready to make a real API call.
7. Inspect extracted fields, confidence estimates, warnings, and JSON export.
8. In **7. Extraction quality**, inspect deterministic status, rule-based
   indicator, issues, and validation summaries.
9. Review issues, edit low-confidence fields, and add session notes.
10. Download original, quality, reviewed, or Markdown exports.
11. In **8. Evaluation**, validate the dataset or run local/mocked fixture tests.
12. Optionally opt in to a small live evaluation after confirming cost and quota.

Opening the app never calls Gemini. The evaluation live tab requires explicit
consent, and full live mode requires additional confirmation.

## 6. Document handling

Supported formats:

- PDF via PyMuPDF
- PNG, JPG/JPEG, and WEBP via Pillow

Validation checks extension, size, file signature, MIME type, and sanitized
filename. PDFs are checked for encryption, password protection, page count,
dimensions, and empty/image-only heuristics. Images are checked for format,
decompression limits, EXIF orientation, transparency, and dimensions.

Preprocessing renders PDF pages at a bounded DPI and normalizes images to PNG
page data within configured pixel and page limits. Failed pages are reported
and excluded instead of being silently substituted. Raw source bytes and
prepared page images remain in the Streamlit session; **Clear all** removes
session state.

## 7. Gemini integration

- SDK: `google-genai` using `google.genai.Client().interactions.create`.
- Configured model: `gemini-3.8-flash`; live response behavior is still pending validation.
- Input: prepared PNG page images in page order, not the original PDF bytes.
- Output: schema-constrained JSON parsed into the existing strict document schemas.
- Continuation: `automatic_continuation=False` to avoid unbounded follow-up calls.
- Retries: SDK retry policy for transient statuses plus bounded correction attempts for malformed output.
- Errors: provider, authentication, quota, network, model, and invalid-response
  paths use safe messages and allowlisted diagnostics.
- Secrets: the key is passed only to the SDK client and is never included in
  UI, logs, exports, or reports.

Confidence values are model estimates, not calibrated probabilities.

## 8. Structured extraction

The provider response is converted to raw text, parsed as JSON, normalized, and
validated against the selected document type. Validation failures can trigger a
bounded correction prompt. The final result stores the original extracted
payload, missing fields, field-level confidence, review flags, validation
errors, and provenance metadata.

Strict structure checks do not prove factual correctness against the source
image. That is why quality assessment and human review remain separate layers.

## 9. Quality assessment

Quality assessment makes no network calls. It assigns a status and a
rule-based indicator from deterministic checks:

- `PASS`: no meaningful issue detected.
- `REVIEW_REQUIRED`: low confidence, inconsistency, type mismatch, or reviewable finding.
- `INCOMPLETE`: missing important content or partial preprocessing.
- `FAILED`: extraction failed, no typed payload, or schema validation failed.

The indicator starts at 100 and deducts by severity, with a floor of 0. It is
not AI accuracy, factual correctness, or probability. Checks cover schema and
processing status, missing/unreadable content, confidence metadata, invoice and
receipt arithmetic, date horizons, note uncertainty, and diagram structure.

Small per-rule supports in the synthetic dataset make per-rule rates explanatory
only.

## 10. Human review and exports

Review state is session-only and never overwrites the original extraction.

- Issue reviews can be marked reviewed or resolved; resolved findings remain visible.
- Field edits are stored separately with original and edited values.
- Reviewer notes and reset controls are available.
- Reviewed output is explicitly labeled as not source-verified.

Exports:

- Original extraction JSON
- Quality JSON with review state
- Reviewed output JSON
- Markdown quality report
- Evaluation JSON, Markdown, and CSV reports

Exports are JSON-serialized and exclude API keys, raw provider payloads, image
bytes, and base64 document data. Nested reviewed-field paths create the needed
containers so a missing path does not silently drop an edit.

## 11. Evaluation framework

The dataset under `evaluation/` contains 11 synthetic samples: four in the
small clean subset and seven additional edge cases in the full subset. Ground
truth is manually authored and never generated or approved by Gemini.

Modes are always labeled:

- `local_synthetic_tests`: deterministic fixture provider, no network.
- `mocked_api_tests`: mocked provider path, no network.
- `real_gemini_api_evaluation`: explicit opt-in, billable, quota-consuming.

Reports include schema validity, exact and normalized field accuracy, missing
and extra fields, document-type accuracy, per-type metrics, quality-rule
precision/recall/F1, review flags, performance, error analysis, limitations,
and recommendations. No single score hides failures.

```powershell
py -m src.evaluation.cli validate
py -m src.evaluation.cli run --mode local --subset small
py -m src.evaluation.cli run --mode mock --subset full
```

Live mode requires `--confirm-live`; full live mode also requires
`--confirm-full`. Dataset sample paths are containment-checked under
`evaluation/samples/`.

## 12. Configuration

| Setting | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | empty | Enables extraction; never log or commit |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Model used for extraction |
| `VISION_PROVIDER` | `gemini` | Only implemented provider |
| `MAX_RETRIES` | 2 | Shared bound for SDK retry attempts and local correction attempts |
| `API_TIMEOUT_SECONDS` | 60 | Per-call timeout |
| `TEMPERATURE` | 0 | Generation temperature |
| `MAX_OUTPUT_TOKENS` | 8192 | Response token limit |
| `CONFIDENCE_REVIEW_THRESHOLD` | 0.75 | Review cutoff |
| `MAX_UPLOAD_SIZE_MB` | 20 | Per-file limit |
| `MAX_DOCUMENTS_PER_SESSION` | 20 | Session document limit |
| `MAX_PAGES_TO_PROCESS` | 10 | Default rendered page limit |
| `PDF_RENDER_DPI` | 110 | Rendering resolution |
| `VISION_MAX_DIMENSION` | 1536 | Processed image bound |
| `EVALUATION_MAX_WORKERS` | 1 | Bound on live evaluation concurrency |
| `EVALUATION_REQUEST_DELAY_SECONDS` | 1 | Delay between live calls |

## 13. Security and privacy

- Uploaded files and prepared page images stay in session memory by default.
- Prepared page images are transmitted to Gemini only when extraction starts.
- Local evaluation and mocked evaluation make no network calls.
- API key status is display-safe; the value is never shown.
- Provider diagnostics sanitize keys, bearer tokens, long base64 blobs, and request IDs.
- Generic UI/CLI failures avoid raw exception text and stack traces.
- Dataset sample filenames must remain under `evaluation/samples/`.
- Exports contain no document bytes, keys, or raw provider payloads.
- `.env` is ignored; `.env.example` is the safe template.
- A local credential-like key exists in `.env`; rotate it if the machine or
  backup has ever been shared.
- Persisted evaluation artifacts should be treated as sensitive after any live run.

## 14. Architecture and project structure

```text
app.py                    Streamlit interface and session orchestration
src/
  config.py               Validated environment configuration
  document/               Validation, ingestion, preprocessing, session state
  extraction/             Provider, parser, prompts, service, exports
  quality/                Deterministic assessment, rules, review, exports
  evaluation/             Dataset, fixtures, runner, metrics, reports, CLI
  schemas/                Strict Pydantic models
  utils/                  Logging
evaluation/               Dataset, ground truth, results, reports
tests/                    Hermetic unit and AppTest coverage
docs/                     Focused architecture, user, development, portfolio docs
```

Detailed boundaries are in `docs/architecture.md`.

## 15. Verification and repository readiness

Local verification commands:

```powershell
py -m pytest -q
py -m compileall -q src app.py tests
py -m pip check
py -m src.evaluation.cli validate
py -m streamlit run app.py
```

For a non-interactive health check, start Streamlit headless and request
`/_stcore/health`; the prior audit returned `200 ok`.

Documentation map:

- `docs/architecture.md`: component boundaries and data flow
- `docs/user_guide.md`: end-to-end operator workflow
- `docs/development.md`: setup, tests, and safe change workflow
- `docs/portfolio_project_summary.md`: evidence-backed project narrative
- `evaluation/README.md`: dataset and metric definitions
- `README.md` section 2 and `docs/portfolio_project_summary.md`: current status and validation limitations

Repository publication checklist:

1. Confirm no `.env`, credentials, caches, or raw live artifacts are staged.
2. Confirm the `LICENSE` copyright holder placeholder before publishing.
3. Keep generated result/report directories out of the initial public commit unless sanitized.
4. Add optional CI for tests, compile, dependency check, and dataset validation.
5. Initialize Git and commit only with separate explicit approval.

## 16. Limitations, future work, and portfolio framing

Current limitations:

- This is not traditional OCR; extraction depends on Gemini vision output.
- Live Gemini extraction and Pydantic validation on a real response are unverified.
- Four-document and full live evaluations are incomplete.
- Confidence is model-estimated, not calibrated.
- Quality status and score are deterministic heuristics.
- Strict schemas validate structure and types, not truth.
- Review state is session-only; refresh clears it.
- The dataset is small, synthetic, and limited to four document types.
- PDF empty/image-only labels use embedded-object heuristics.
- Encrypted PDFs are rejected rather than decoded.
- There is no persistence, authentication, deployment, or CI here.
- The MIT `LICENSE` file is present; its copyright holder placeholder needs
  confirmation before publication.

Prioritized future work:

1. Obtain one successful controlled live Gemini response.
2. Confirm live parsing and Pydantic validation.
3. Run the four-document live evaluation, then decide on full live evaluation.
4. Confirm the `LICENSE` copyright holder, then add repository packaging and minimal CI.
5. Consider optional persistence, authentication, and deployment only as separate approved phases.

Portfolio framing: describe this as a locally complete multimodal extraction
workflow with a real Gemini integration implemented and live validation pending
because of provider capacity failures. Do not claim real-world accuracy,
production deployment, or infrastructure that does not exist.
