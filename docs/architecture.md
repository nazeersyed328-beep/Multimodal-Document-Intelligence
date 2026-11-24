# Architecture

## Scope

This document describes component boundaries and data flow. Setup commands and
feature status live in `README.md`; operator steps live in `docs/user_guide.md`.

## Component boundaries

| Layer | Owner | Responsibility |
| --- | --- | --- |
| Interface | `app.py` | Streamlit pages, session state keys, user actions, safe display |
| Configuration | `src/config.py` | Parse environment, validate values, expose display-safe summary |
| Document | `src/document/` | Validate uploads, inspect PDFs/images, render pages, bound session data |
| Extraction | `src/extraction/` | Provider-neutral calls, Gemini SDK adapter, JSON parsing, schema validation, exports |
| Quality | `src/quality/` | Deterministic issue rules, score, review state, reviewed output |
| Evaluation | `src/evaluation/` | Dataset, fixtures, runners, comparison, metrics, reports, CLI |
| Schemas | `src/schemas/` | Strict Pydantic models and shared enums |
| Logging | `src/utils/` | Console logging with secret/body redaction |

## Main data flow

1. `app.py` receives uploaded bytes and calls `ingest_uploads`.
2. `src/document/validators.py` checks extension, size, signature, MIME, and filename.
3. `src/document/ingestion.py` creates a `DocumentRecord` with metadata and SHA-256.
4. `src/document/preprocessing.py` renders PDF pages or normalizes images into
   bounded PNG page bytes.
5. `ExtractionService` builds a prompt and schema, then calls the configured
   `VisionProvider`.
6. `GeminiVisionProvider` sends prepared page images and receives raw text.
   It never logs the key, images, or full SDK response.
7. `src/extraction/parser.py` parses JSON and fills strict document schemas.
8. `QualityAssessmentService` compares extraction metadata and deterministic
   rules, producing a `QualityReport`.
9. Review helpers store edits and notes separately; exports never mutate the
   original extraction.
10. Evaluation runners reuse the same pipeline with either fixture providers or
    an explicitly opted-in live provider.

## State boundaries

- Streamlit session state owns uploaded records, preprocessing results,
  extraction results, quality reports, review states, and evaluation summaries.
- Nothing is persisted by the UI. Browser refresh creates a new session.
- Evaluation raw results and reports are written under `evaluation/` when a run
  is started; local/mocked content is synthetic, live content must be treated as sensitive.
- Environment configuration is loaded from `.env` unless an explicit mapping is
  supplied to `load_settings`.

## Safety boundaries

- Provider errors are mapped to allowlisted exception classes and diagnostics.
- `sanitize_diagnostic_text` removes common key, bearer-token, base64, and
  assignment patterns from diagnostic text.
- Dataset sample filenames are resolved and checked against the samples root.
- Review field paths use next-token-aware container creation for nested edits.
- Unexpected provider response shapes are rejected instead of being treated as
  empty valid output.
- App and CLI unexpected failures display exception type rather than raw text.
- Live API usage requires explicit action in the UI or CLI confirmation flags.

## Evaluation modes

| Mode | Provider | Network | Purpose |
| --- | --- | --- | --- |
| `local_synthetic_tests` | Deterministic fixture | No | Pipeline and metric verification |
| `mocked_api_tests` | Same fixture, separate label | No | Mock API orchestration verification |
| `real_gemini_api_evaluation` | Gemini provider | Yes | Opt-in live measurement; currently blocked by `503` failures |

Reports keep mode, dataset version, model ID, and api status so synthetic runs
cannot be mistaken for live model accuracy.

## Extension points

- New document schemas: add under `src/schemas/` and wire parser/service mappings.
- New provider: implement `VisionProvider`; configuration currently allows only Gemini.
- New quality rules: add deterministic rules under `src/quality/` with tests.
- New samples: add synthetic artwork, manifest entries, and manually authored ground truth.

## Non-goals

Traditional OCR, authentication, database persistence, deployment, multi-provider
support, real-world accuracy claims, and autonomous report generation from model
answers are outside the current system boundary.
