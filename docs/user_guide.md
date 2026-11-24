# User Guide

## Before starting

- Open the Streamlit app from the project root with `py -m streamlit run app.py`.
- Local upload, inspection, preview, review, and fixture evaluation do not need a key.
- Real extraction and live evaluation require `GEMINI_API_KEY` and make billable calls.
- A healthy page load does not call Gemini; only explicit buttons do.

## Recommended first session

1. Upload one synthetic PDF or image from `evaluation/samples/`.
2. Confirm the validation table shows the expected format, size, pages, and status.
3. Expand document metadata to inspect sanitized filename, MIME type, hash, and PDF/image details.
4. Process the document and read preprocessing messages before extracting.
5. Preview the prepared page and confirm the visible content is usable.
6. Choose **Auto-detect** unless the document type is known.
7. Start extraction only when ready to send prepared page images to Gemini.
8. Review quality and exports before drawing conclusions.

## Reading extraction output

- **Overall confidence** and field confidence are model estimates, not probabilities.
- **Needs review** means the pipeline found uncertainty or inconsistency; it does
  not prove the value is wrong.
- Missing fields, unreadable fields, low confidence, and arithmetic issues are
  recorded explicitly instead of being guessed.
- Download the original extraction JSON to preserve the untouched model output.

## Reading quality output

- Quality status is a coarse deterministic classification.
- Rule-based indicator is a heuristic score, not accuracy.
- Detected issues remain in the report even after being marked resolved.
- Issue review records a human action; it does not verify the source image.
- Reviewed field values are stored separately from the original extraction.

## Review and export

1. Open an issue row to see rule ID, severity, affected fields, expected/actual values, and recommendation.
2. Set review status and note, then save.
3. Open a low-confidence field to compare AI value, reason, evidence, and confidence.
4. Save a corrected value only when the source supports it.
5. Add a short session note if context is needed.
6. Download the needed export:
   - original extraction JSON for untouched output;
   - quality JSON for report plus review state;
   - reviewed output JSON for session edits with explicit labels;
   - Markdown report for human-readable review.

Refreshing the page clears review edits. Download anything you need before leaving.

## Evaluation

- Use the **Dataset** tab to validate ground truth and sample hashes.
- Use **Local synthetic** or **Mocked API** for deterministic checks; neither calls Gemini.
- Use **Live Gemini** only after reading the cost warning and checking consent.
- Start with the small subset. Full live mode requires an additional confirmation.
- Reports are labeled by mode. Synthetic success is not real-model accuracy.

## Troubleshooting

| Symptom | Meaning | Safe response |
| --- | --- | --- |
| File rejected | Extension, size, signature, MIME, or filename check failed | Check the listed validation reason and try another file |
| Extraction disabled | No configured key | Add key to `.env`, restart app |
| Preview unavailable | Page could not be rendered/decoded | Try another file or page; no raw exception is shown |
| Provider unavailable | Gemini returned a safe mapped error | Wait and retry later; do not change model as a workaround |
| Empty or unexpected response | Provider response shape was not usable | Treat as a provider failure, not a local parsing success |
| Live validation still pending | No successful live response has been observed | See `README.md` section 2 and `docs/portfolio_project_summary.md` |

## Limits to remember

- Encrypted PDFs are rejected.
- Image-only/empty PDF labels are heuristic.
- Review edits are session-only.
- There is no database, login, or hosted deployment.
- Live extraction quality has not been validated because of repeated provider `503` responses.
