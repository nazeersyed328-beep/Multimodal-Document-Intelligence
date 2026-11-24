# Evaluation dataset

This directory contains a small, repeatable benchmark for the document-intelligence pipeline.

**Status:** local and mocked evaluation are implemented and locally verified.
Live Gemini evaluation is implemented but pending because five controlled live
smoke attempts returned HTTP `503 service_unavailable`. Synthetic results never
represent measured real-model accuracy.

## Provenance

- Dataset version: `phase5-synthetic-v1`.
- Inputs are synthetic images and one synthetic PDF generated locally by `generate_samples.py`.
- Ground truth is manually authored for that synthetic artwork by the artwork author.
- Gemini did not generate, verify, or approve ground truth.
- License for these synthetic assets: `CC0-1.0 for these synthetic assets`.
- `samples/manifest.json` records file sizes and SHA-256 hashes.
- `index.json` records dataset version, sample count, source, and ground-truth method.

## Sample inventory

| Sample | Type | Subset | Purpose |
| --- | --- | --- | --- |
| `inv-clean-001.png` | invoice | small | Clean control |
| `inv-missing-001.png` | invoice | full | Missing total/tax fields; primary expected rules: `GEN-007`, `GEN-012`, `INV-LINE-005` |
| `inv-arith-001.png` | invoice | full | Inconsistent arithmetic; primary expected rules: `INV-TOTAL-002`, `GEN-012` |
| `inv-multipage-001.pdf` | invoice | full | Two-page PDF control |
| `rcpt-clean-001.png` | receipt | small | Clean control |
| `rcpt-blurry-001.png` | receipt | full | Blurred source; expected low-confidence fields |
| `rcpt-future-001.png` | receipt | full | Future transaction date; expects `DOC-DATE-002` |
| `note-clean-001.png` | handwritten_note | small | Clean control |
| `note-unreadable-001.png` | handwritten_note | full | Unreadable spans; expects `NOTE-002`, `NOTE-006` |
| `diag-clean-001.png` | diagram | small | Clean control |
| `diag-ambiguous-001.png` | diagram | full | Ambiguous connections; expects `DIAG-006`, `DIAG-010` |

The small subset contains one clean sample per document type. The full subset contains all 11 samples.

## Commands

From the project root:

```powershell
py evaluation\generate_samples.py
py -m src.evaluation.cli validate
py -m src.evaluation.cli run --mode local --subset small
py -m src.evaluation.cli run --mode local --subset full
py -m src.evaluation.cli run --mode mock --subset small
py -m src.evaluation.cli run --mode live --subset small --confirm-live
py -m src.evaluation.cli run --mode live --subset full --confirm-live --confirm-full
```

Opening the Streamlit app never starts an API evaluation. Use section **8. Evaluation** and explicitly press a run button.

## Modes

- `local_synthetic_tests`: deterministic fixture provider. No network calls. Verifies dataset, ingestion, preprocessing, extraction orchestration, quality rules, comparison, metrics, and reporting.
- `mocked_api_tests`: same fixture path, labeled separately to test mock API orchestration. No network calls.
- `real_gemini_api_evaluation`: uses the configured Gemini provider. Requires `--confirm-live`, a configured key, small-subset first, and `--confirm-full` for full mode.

## Metric definitions

- Schema validity: fraction of documents with a typed payload and `validation_status != invalid`. `needs_review` is still schema-valid.
- Exact-match accuracy: fraction of compared field values that match after JSON-safe serialization.
- Normalized accuracy: fraction of compared field values that match after NFKC normalization, trimming, whitespace collapsing, casefolding, numeric tolerance, or ISO date comparison.
- Missing-field rate: missing fields divided by fields with a non-null expectation.
- Incorrectly populated rate: incorrect plus extra fields divided by all compared fields.
- Document-type accuracy: fraction of documents whose detected type equals the expected type.
- Quality-rule precision/recall/F1: micro-averaged comparison of expected and detected rule IDs; per-rule metrics are also reported.
- Review-required metrics: binary comparison of expected and actual review flags.
- Low-confidence and unreadable-field metrics: field-path set comparisons; supports are small and explanatory only.
- API request success rate: documents with no recorded error code divided by documents evaluated.
- Latency: wall-clock time from document load through quality assessment, including all provider calls.

`None` is reported as `not measurable` when a denominator is zero. No metric is imputed or fabricated.

## Outputs

- `results/<run_key>/<sample_id>.json`: raw extraction result, quality report, and evaluation record.
- `results/<run_key>/summary.json`: complete machine-readable run summary.
- `reports/*_summary.json`: report JSON with report schema version.
- `reports/*_report.md`: 14-section Markdown report.
- `reports/*_errors.csv`: row-level error analysis.

Resume reuses a raw result only when its fingerprint matches the dataset sample, ground truth, mode, model ID, confidence threshold, tolerance, date settings, and quality limits. Live evaluation uses resume by default; local and mocked CLI runs do not.

## Adding samples

1. Add synthetic artwork to `samples/` and record it in `samples/manifest.json`.
2. Add a manually authored ground-truth JSON file in `ground_truth/`.
3. Use a unique `sample_id`, the exact `file_name` contained under `samples/`, correct `document_type`, dataset version, source, license, `checked_by`, and `checked_at`. Paths outside the samples root are rejected.
4. Mark `dataset_subset` as `small` only for clean control samples.
5. Run `py -m src.evaluation.cli validate` before evaluating.
6. Never copy ground truth from a model answer and never fabricate a result.

## Limitations

- Synthetic documents are cleaner and more regular than many real-world scans.
- Only 11 samples and four document types are covered.
- Some expected issue rules have support of one, so precision and recall are not statistically stable.
- Model confidence is not calibrated probability.
- The quality score remains a deterministic rule-based indicator, not AI accuracy.
- Live evaluation is bounded and delayed by default, but API cost and quota still apply.
- Five controlled live smoke attempts returned `503`; no live response or live accuracy metric exists yet.
