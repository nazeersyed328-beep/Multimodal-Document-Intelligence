# Development Guide

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Use `py -m ...` for all Python commands on this Windows workspace. Keep real
keys in `.env` only, never in source, tests, reports, or commits.

## Verification commands

Run from the project root:

```powershell
py -m pytest -q
py -m compileall -q src app.py tests
py -m pip check
py -m src.evaluation.cli validate
py -m src.evaluation.cli run --mode local --subset small
```

For a non-mutating interface health check, start Streamlit headless and request
`/_stcore/health`; expect `200 ok`. Terminate the process after the check.

The live smoke test is skipped by default:

```powershell
$env:RUN_LIVE_GEMINI_TESTS="1"
py -m pytest -v tests/test_live_gemini.py
```

Do not enable it during offline finalization work. It requires separate
approval because it makes real API calls.

## Test layout

| Area | Tests |
| --- | --- |
| Config and schemas | `tests/test_config.py`, `tests/test_schemas.py` |
| Document pipeline | `tests/test_document_validators.py`, `tests/test_ingestion.py`, `tests/test_preprocessing.py`, `tests/test_session.py` |
| Extraction | `tests/test_extraction_*.py` |
| Quality and review | `tests/test_quality_*.py` |
| Evaluation | `tests/test_evaluation.py` |
| Streamlit | `tests/test_app.py` |
| Optional live smoke | `tests/test_live_gemini.py` |

## Change conventions

- Keep provider-neutral logic in `src/extraction/provider.py`; put SDK-specific code in `gemini_provider.py`.
- Add tests for any parsing, path, export, or error-handling change.
- Preserve mode labels in evaluation reports; never let a fixture result look live.
- Use sanitized diagnostics and generic user messages for unexpected failures.
- Keep document bytes, API keys, base64, and full SDK objects out of logs and errors.
- Use `.env.example` for new settings and update `Settings.public_summary()` for display-safe values.
- Keep README status honest: live validation remains pending until a successful real response exists.

## Safe API work

- Mock provider calls in normal tests with the fake client pattern in `tests/test_extraction_provider.py`.
- Controlled smoke runs should use exactly one request and `MAX_RETRIES=0`.
- Never print environment variables, auth headers, payloads, image bytes, or full SDK objects.
- Five recorded smoke attempts returned `503 service_unavailable`; repeated immediate retries add cost without new evidence.
- Do not change model, SDK, endpoint, or request format as an undocumented workaround.

## Adding evaluation samples

1. Add synthetic artwork under `evaluation/samples/`.
2. Update `evaluation/samples/manifest.json` with size and SHA-256.
3. Add manually authored ground truth under `evaluation/ground_truth/`.
4. Use a unique sample ID, exact filename, source, license, checker, and date.
5. Update `evaluation/index.json` counts/version only with a deliberate dataset version change.
6. Run `py -m src.evaluation.cli validate`.
7. Never copy ground truth from a model answer.

## Publication workflow

1. Finish documentation and local regression checks.
2. Scan for secrets without printing values.
3. Confirm `.env`, caches, and unwanted generated outputs are ignored.
4. Confirm the LICENSE copyright holder and decide CI separately.
5. Initialize Git, stage, and commit only with explicit approval.
6. Push only with separate explicit approval.

## Current blockers

- No successful live Gemini response.
- No live parsing/Pydantic validation evidence.
- No four-document or full live evaluation.
- No Git repository, CI, auth, persistence, or deployment; the LICENSE copyright holder placeholder needs confirmation.

Production readiness requires successful live validation; offline finalization
does not remove that blocker.
