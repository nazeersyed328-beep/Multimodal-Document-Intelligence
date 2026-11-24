"""Phase 5 Streamlit application for document evaluation and benchmarking.

The app accepts uploads, validates and inspects them locally, renders previews,
prepares normalized page images in memory, extracts schema-validated data through
Gemini, adds deterministic quality assessment, and runs labeled evaluations.
"""

from __future__ import annotations

import importlib.util
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import streamlit as st

from src import __version__
from src.config import ConfigurationError, PROJECT_ROOT, Settings, get_settings
from src.document.ingestion import DocumentRecord, ingest_uploads
from src.document.preprocessing import (
    make_preview_from_bytes,
    preprocess_document,
    render_pdf_page,
)
from src.document.session import DocumentSession
from src.evaluation.dataset import (
    DEFAULT_EVALUATION_ROOT,
    load_dataset,
    select_samples,
    validate_dataset,
)
from src.evaluation.fixtures import FixtureVisionProvider
from src.evaluation.reporting import render_markdown_report, write_reports
from src.evaluation.runner import EvaluationRunner
from src.evaluation.schemas import (
    EvaluationMode,
    EvaluationRunSummary,
    EvaluationSubset,
)
from src.extraction import ExtractionService, dump_extraction_result
from src.extraction.provider import (
    InvalidResponseError,
    ProviderError,
    sanitize_diagnostic_text,
)
from src.quality import (
    QualityAssessmentService,
    apply_review_to_report,
    build_reviewed_output,
    dump_original_extraction,
    dump_quality_report,
    dump_reviewed_output,
    new_review_state,
    remove_field_edit,
    remove_issue_review,
    render_quality_markdown,
    reset_review,
    review_progress,
    set_field_edit,
    set_issue_review,
    add_reviewer_note,
)
from src.schemas.common import DocumentType
from src.schemas.document_ingestion import DocumentFormat, PreprocessingResult
from src.schemas.extraction_result import ExtractionResult, ProcessingStatus
from src.schemas.quality import (
    IssueReviewStatus,
    QualityCategory,
    QualityReport,
    QualitySeverity,
    ReviewState,
)
from src.utils.logging_config import configure_logging, get_logger

SESSION_KEY = "phase2_document_session"
UPLOADER_KEY = "phase2_uploader"
EXTRACTION_KEY = "phase3_extraction_results"
QUALITY_KEY = "phase4_quality_reports"
REVIEW_KEY = "phase4_review_states"
DETECTED_TYPE_KEY = "phase4_detected_types"
EVALUATION_SUMMARY_KEY = "phase5_evaluation_summary"
EVALUATION_REPORTS_KEY = "phase5_evaluation_reports"

DOCUMENT_TYPE_OPTIONS: tuple[DocumentType | None, ...] = (
    None,
    DocumentType.INVOICE,
    DocumentType.RECEIPT,
    DocumentType.HANDWRITTEN_NOTE,
    DocumentType.DIAGRAM,
    DocumentType.UNKNOWN,
)
DOCUMENT_TYPE_LABELS = {
    None: "Auto-detect",
    DocumentType.INVOICE: "Invoice",
    DocumentType.RECEIPT: "Receipt",
    DocumentType.HANDWRITTEN_NOTE: "Handwritten note",
    DocumentType.DIAGRAM: "Diagram",
    DocumentType.UNKNOWN: "Unknown (skip extraction)",
}

REQUIRED_MODULES = {
    "pydantic": "pydantic",
    "streamlit": "streamlit",
    "python-dotenv": "dotenv",
    "Pillow": "PIL",
    "PyMuPDF": "pymupdf",
    "google-genai": "google.genai",
    "pytest": "pytest",
}
REQUIRED_DIRS = (
    "src",
    "src/schemas",
    "src/document",
    "src/extraction",
    "src/quality",
    "src/utils",
    "tests",
    "outputs",
    "reports",
    "sample_documents",
)


def _module_is_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _upload_id(upload: object) -> str | None:
    value = getattr(upload, "file_id", None)
    return str(value) if value is not None else None


def _get_session() -> DocumentSession:
    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = DocumentSession()
    return st.session_state[SESSION_KEY]


def _get_extractions() -> dict[str, ExtractionResult]:
    if EXTRACTION_KEY not in st.session_state:
        st.session_state[EXTRACTION_KEY] = {}
    return st.session_state[EXTRACTION_KEY]


def _get_quality_reports() -> dict[str, QualityReport]:
    if QUALITY_KEY not in st.session_state:
        st.session_state[QUALITY_KEY] = {}
    return st.session_state[QUALITY_KEY]


def _get_review_states() -> dict[str, ReviewState]:
    if REVIEW_KEY not in st.session_state:
        st.session_state[REVIEW_KEY] = {}
    return st.session_state[REVIEW_KEY]


def _get_detected_types() -> dict[str, DocumentType | None]:
    if DETECTED_TYPE_KEY not in st.session_state:
        st.session_state[DETECTED_TYPE_KEY] = {}
    return st.session_state[DETECTED_TYPE_KEY]


def _review_key(result: ExtractionResult) -> str:
    return result.source_document_id or result.result_id


def _extraction_label(record: DocumentRecord) -> str:
    result = _get_extractions().get(record.document_id)
    if result is None:
        return "not extracted"
    report = _get_quality_reports().get(record.document_id)
    if report is not None:
        return f"extracted - {report.quality_status.value}"
    if result.needs_review:
        return "needs review"
    return result.status.value


def _short(value: object, limit: int = 120) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _display_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(f"{key}={item}" for key, item in value.items())
    return str(value)


def _coerce_review_value(original: object, text: str) -> object:
    cleaned = text.strip()
    if original is None:
        return cleaned or None
    if isinstance(original, bool):
        lowered = cleaned.lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
        return cleaned
    if isinstance(original, int):
        try:
            return int(cleaned)
        except ValueError:
            return cleaned
    if isinstance(original, Decimal):
        try:
            return Decimal(cleaned)
        except InvalidOperation:
            return cleaned
    if isinstance(original, float):
        try:
            return float(cleaned)
        except ValueError:
            return cleaned
    if isinstance(original, datetime):
        try:
            return datetime.fromisoformat(cleaned)
        except ValueError:
            return cleaned
    if isinstance(original, date):
        try:
            return date.fromisoformat(cleaned)
        except ValueError:
            return cleaned
    if isinstance(original, (list, tuple)):
        return [item.strip() for item in cleaned.split(",") if item.strip()]
    return cleaned or None


def _safe_key(*parts: object) -> str:
    raw = "_".join(str(part) for part in parts)
    return "".join(character if character.isalnum() or character in "_-" else "_" for character in raw)


def _page_count(record: DocumentRecord) -> int:
    if record.metadata.pdf is not None:
        return record.metadata.pdf.page_count
    return 1


def _status_label(session: DocumentSession, record: DocumentRecord) -> str:
    result = session.processed_for(record.document_id)
    if result is None:
        return "Validated"
    return {
        ProcessingStatus.PENDING: "Processing",
        ProcessingStatus.IN_PROGRESS: "Processing",
        ProcessingStatus.COMPLETED: "Processed",
        ProcessingStatus.REQUIRES_REVIEW: "Processed with issues",
        ProcessingStatus.FAILED: "Failed",
    }.get(result.status, "Unknown")


def _render_upload_section(session: DocumentSession, settings: Settings) -> None:
    st.subheader("1. Upload documents")
    st.caption(
        f"Accepted: {', '.join(settings.allowed_file_extensions)} | "
        f"Maximum per file: {settings.max_upload_size_mb} MB | "
        f"Maximum documents per session: {settings.max_documents_per_session}"
    )
    uploads = st.file_uploader(
        "Choose one or more documents",
        type=[extension.lstrip(".") for extension in settings.allowed_file_extensions],
        accept_multiple_files=True,
        key=UPLOADER_KEY,
        max_upload_size=settings.max_upload_size_bytes,
        help=(
            "Uploaded files stay in memory for this session. Prepared page images are "
            "sent to Gemini only when you start extraction."
        ),
    )
    if not uploads:
        return

    new_uploads = [upload for upload in uploads if not session.is_upload_seen(_upload_id(upload))]
    if not new_uploads:
        st.caption("These files have already been added in this session.")
        return

    with st.spinner("Validating and inspecting uploaded files..."):
        batch = ingest_uploads(new_uploads, settings, session.fingerprints)
    for upload in new_uploads:
        session.mark_upload_seen(_upload_id(upload))
    for record in batch.records:
        session.add_record(record)
    session.rejected = list(batch.rejected)
    removed = session.enforce_limit(settings.max_documents_per_session)

    accepted_col, rejected_col = st.columns(2)
    with accepted_col:
        if batch.records:
            st.success(f"Accepted {len(batch.records)} document(s).")
            for record in batch.records:
                st.caption(f"- {record.sanitized_filename}")
        else:
            st.caption("No new valid documents in this batch.")
    with rejected_col:
        if batch.rejected:
            st.error(f"Rejected {len(batch.rejected)} file(s):")
            for result in batch.rejected:
                st.caption(f"**{result.original_filename}**")
                for issue in result.errors:
                    st.caption(f"- {issue.message}")
        if batch.duplicates:
            st.warning(f"Skipped duplicates: {', '.join(batch.duplicates)}")
        if batch.skipped_limit:
            st.warning(
                "Session document limit reached; skipped: "
                + ", ".join(batch.skipped_limit)
            )
    if removed:
        st.warning(f"Removed {len(removed)} oldest document(s) to enforce the session limit.")
    if st.button("Dismiss validation messages", key="dismiss_messages"):
        session.rejected = []
        st.rerun()


def _render_document_list(session: DocumentSession) -> None:
    st.subheader("2. Document list")
    if not session.documents:
        st.info("No validated documents yet. Upload a PDF or image above.")
        return
    rows = [
        {
            "Filename": record.sanitized_filename,
            "Format": record.document_format.value.upper(),
            "Size": _format_bytes(record.size_bytes),
            "Pages": _page_count(record),
            "Status": _status_label(session, record),
            "Extraction": _extraction_label(record),
            "Validation": (
                "warnings" if record.warnings else "passed"
            ),
        }
        for record in session.documents.values()
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_metadata(record: DocumentRecord) -> None:
    with st.expander("Document metadata", expanded=False):
        metadata = record.metadata
        details = {
            "original_filename": metadata.original_filename,
            "sanitized_filename": metadata.sanitized_filename,
            "format": metadata.document_format.value,
            "mime_type": metadata.mime_type,
            "size_bytes": metadata.size_bytes,
            "sha256": metadata.sha256,
        }
        if metadata.image is not None:
            details.update(
                {
                    "width": metadata.image.width,
                    "height": metadata.image.height,
                    "mode": metadata.image.mode,
                    "has_transparency": metadata.image.has_transparency,
                    "exif_orientation": metadata.image.exif_orientation,
                }
            )
        if metadata.pdf is not None:
            inspection = metadata.pdf
            details.update(
                {
                    "page_count": inspection.page_count,
                    "encrypted": inspection.is_encrypted,
                    "needs_password": inspection.needs_password,
                    "empty_pages": [
                        page.page_number
                        for page in inspection.pages
                        if page.content_status.value == "empty"
                    ],
                }
            )
            for key, value in inspection.metadata.model_dump().items():
                if value:
                    details[f"pdf_{key}"] = value
        st.json(details)
        for warning in record.warnings:
            st.caption(f"Note: {warning}")


def _preview_for_record(
    record: DocumentRecord,
    settings: Settings,
    page_number: int,
) -> tuple[bytes, str]:
    result = (
        render_pdf_page(record.raw_bytes, page_number, settings)
        if record.document_format is DocumentFormat.PDF
        else None
    )
    source = (
        result.data
        if result is not None
        else make_preview_from_bytes(record.raw_bytes, settings.vision_max_dimension)
    )
    label = (
        f"Rendered page {page_number} at {settings.pdf_render_dpi:g} DPI"
        if record.document_format is DocumentFormat.PDF
        else "Uploaded image"
    )
    return make_preview_from_bytes(source, settings.preview_max_dimension), label


def _render_preview(
    session: DocumentSession,
    settings: Settings,
    record: DocumentRecord,
) -> None:
    st.subheader("5. Preview")
    result = session.processed_for(record.document_id)
    available_pages = (
        [page for page in result.pages if page.image_bytes] if result is not None else []
    )

    if available_pages:
        page_numbers = [page.page_number for page in available_pages]
        selected_page = st.selectbox(
            "Preview page",
            options=page_numbers,
            key="preview_page_select",
            format_func=lambda number: f"Page {number}",
        )
        page = next(
            (page for page in available_pages if page.page_number == selected_page),
            available_pages[0],
        )
        preview = make_preview_from_bytes(page.image_bytes, settings.preview_max_dimension)
        caption = (
            f"Processed page {page.page_number} - "
            f"{page.processed_width}x{page.processed_height} px"
        )
        if preview:
            st.image(preview, caption=caption, use_container_width=True)
        if page.warnings:
            for warning in page.warnings:
                st.caption(f"Note: {warning}")
    else:
        total_pages = _page_count(record)
        selected_page = 1
        if total_pages > 1:
            selected_page = st.selectbox(
                "Preview page",
                options=list(range(1, total_pages + 1)),
                key="preview_page_select",
                format_func=lambda number: f"Page {number}",
            )
        try:
            preview, caption = _preview_for_record(record, settings, selected_page)
        except Exception as exc:
            st.error(f"Preview unavailable: {type(exc).__name__}")
            return
        if preview:
            st.image(preview, caption=caption, use_container_width=True)
        else:
            st.warning("No preview could be generated for this document.")

    _render_metadata(record)


def _render_processing_results(result: PreprocessingResult) -> None:
    st.subheader("4. Preprocessing status")
    if result.status is ProcessingStatus.COMPLETED:
        st.success(f"Processed {len(result.pages)} page(s) successfully.")
    elif result.status is ProcessingStatus.REQUIRES_REVIEW:
        st.warning("Processed with page failures. Review the messages below.")
    else:
        st.error("Preprocessing failed for this document.")
    for error in result.errors:
        st.caption(f"Error: {error}")
    for warning in result.warnings:
        st.caption(f"Note: {warning}")
    st.caption(f"Processing time: {result.processing_time_seconds:.3f} seconds.")
    with st.expander("Prepared page details", expanded=False):
        for page in result.pages:
            st.json(
                {
                    "page": page.page_number,
                    "status": page.processing_status.value,
                    "original_dimensions": [page.original_width, page.original_height],
                    "processed_dimensions": [
                        page.processed_width,
                        page.processed_height,
                    ],
                    "format": page.image_format,
                    "sha256": page.sha256,
                    "error": page.error,
                }
            )


def _render_controls(session: DocumentSession, settings: Settings) -> DocumentRecord | None:
    st.subheader("3. Processing controls")
    if not session.documents:
        st.caption("Upload a valid document to enable processing controls.")
        return None

    document_ids = list(session.documents)
    selected_id = st.selectbox(
        "Selected document",
        options=document_ids,
        key="selected_document",
        format_func=lambda doc_id: session.documents[doc_id].sanitized_filename,
    )
    control_cols = st.columns(4)

    with control_cols[0]:
        if st.button("Process selected", key="process_selected"):
            record = session.documents[selected_id]
            with st.spinner("Preparing page images..."):
                result = preprocess_document(record, settings)
            session.set_processed(selected_id, result)
    with control_cols[1]:
        if st.button("Process all valid", key="process_all"):
            with st.spinner("Preparing page images for all documents..."):
                for record in tuple(session.documents.values()):
                    result = preprocess_document(record, settings)
                    session.set_processed(record.document_id, result)
    with control_cols[2]:
        if st.button("Remove selected", key="remove_selected"):
            session.remove(selected_id)
            _get_extractions().pop(selected_id, None)
            _get_quality_reports().pop(selected_id, None)
            _get_review_states().pop(selected_id, None)
            _get_detected_types().pop(selected_id, None)
            st.session_state.pop("preview_page_select", None)
            st.rerun()
    with control_cols[3]:
        if st.button("Clear all", key="clear_all"):
            session.clear()
            _get_extractions().clear()
            _get_quality_reports().clear()
            _get_review_states().clear()
            _get_detected_types().clear()
            st.session_state["phase2_uploader"] = []
            st.session_state.pop("preview_page_select", None)
            st.rerun()
    return session.documents.get(selected_id)


def _render_extraction_result(settings: Settings, result: ExtractionResult) -> None:
    status_text = {
        ProcessingStatus.COMPLETED: "Completed",
        ProcessingStatus.REQUIRES_REVIEW: "Needs review",
        ProcessingStatus.FAILED: "Failed",
        ProcessingStatus.PENDING: "Pending",
        ProcessingStatus.IN_PROGRESS: "In progress",
    }.get(result.status, result.status.value)
    if result.status is ProcessingStatus.FAILED:
        st.error(f"Extraction failed: {status_text}.")
    elif result.needs_review:
        st.warning("Extraction completed, but this document is marked for review.")
    else:
        st.success("Extraction completed.")

    confidence = result.overall_confidence
    st.metric("Overall confidence", "not provided" if confidence is None else f"{confidence:.2f}")
    st.caption(
        f"Provider: {settings.vision_provider} | Model: {settings.gemini_model} | "
        f"Confidence values are model estimates, not calibrated probabilities."
    )

    for error in result.validation_errors:
        location = f" ({error.field_path})" if error.field_path else ""
        st.caption(f"Error{location}: {error.message}")
    for warning in result.warnings:
        st.caption(f"Note: {warning}")

    if result.data is not None:
        st.markdown("#### Extracted fields")
        st.json(result.data.model_dump(mode="json"))
    else:
        st.info("No structured data was returned for this document.")

    if result.field_confidence:
        st.markdown("#### Field confidence")
        confidence_rows = [
            {
                "Field": field_path,
                "Value": _short(entry.value),
                "Confidence": (
                    "not provided"
                    if entry.confidence is None
                    else f"{entry.confidence:.2f}"
                ),
                "Review": "yes" if entry.needs_review else "no",
                "Evidence": _short(entry.evidence) if entry.evidence else "",
            }
            for field_path, entry in result.field_confidence.items()
        ]
        st.dataframe(confidence_rows, use_container_width=True, hide_index=True)

    st.markdown("#### JSON export")
    json_text = dump_extraction_result(result)
    st.caption("Select the JSON below to copy it, or download it as a file.")
    st.code(json_text, language="json")
    file_stem = Path(result.source_document_id or result.result_id).stem
    st.download_button(
        "Download extraction JSON",
        data=json_text,
        file_name=f"{file_stem}_extraction.json",
        mime="application/json",
        key=f"download_{result.result_id}",
    )


def _render_extraction_section(
    session: DocumentSession,
    settings: Settings,
    selected: DocumentRecord | None,
) -> None:
    st.subheader("6. AI extraction")
    if not settings.is_api_key_configured:
        st.warning(
            "GEMINI_API_KEY is not configured. Uploading, preprocessing, and "
            "previewing still work locally; extraction is disabled."
        )
        return
    if selected is None:
        st.info("Select a valid document above to enable AI extraction.")
        return

    processed = session.processed_for(selected.document_id)
    if processed is None:
        st.info("Process the selected document first to prepare its page images.")
        return
    if not any(
        page.image_bytes and page.processing_status is ProcessingStatus.COMPLETED
        for page in processed.pages
    ):
        st.warning("No prepared page images are available for extraction.")
        return

    selected_type = st.selectbox(
        "Document type",
        options=DOCUMENT_TYPE_OPTIONS,
        key="extraction_document_type",
        format_func=lambda value: DOCUMENT_TYPE_LABELS[value],
    )
    if st.button("Extract with Gemini", key="extract_with_gemini"):
        try:
            service = ExtractionService(settings)
            with st.spinner("Sending prepared pages to Gemini..."):
                result = service.extract_document(selected, processed, selected_type)
            _get_extractions()[selected.document_id] = result
            _get_detected_types()[selected.document_id] = (
                selected_type if selected_type not in (None, DocumentType.UNKNOWN) else None
            )
            _get_quality_reports().pop(selected.document_id, None)
            _get_review_states().pop(selected.document_id, None)
        except (ProviderError, InvalidResponseError) as exc:
            st.error(exc.user_message)
        except ValueError as exc:
            detail = sanitize_diagnostic_text(exc) or type(exc).__name__
            st.error(f"Extraction input error: {detail}")
        except Exception as exc:
            st.error(f"Extraction failed ({type(exc).__name__}).")

    existing = _get_extractions().get(selected.document_id)
    if existing is not None:
        _render_extraction_result(settings, existing)
        _render_quality_section(settings, processed, existing)


def _current_quality_report(
    settings: Settings,
    preprocessing: PreprocessingResult | None,
    result: ExtractionResult,
) -> QualityReport:
    reports = _get_quality_reports()
    cached = reports.get(result.source_document_id or "")
    if cached is not None:
        return cached
    report = QualityAssessmentService(settings).assess(
        result,
        preprocessing=preprocessing,
        detected_type=_get_detected_types().get(result.source_document_id or ""),
    )
    reports[result.source_document_id or ""] = report
    return report


def _render_issue_review_controls(
    report: QualityReport,
    result: ExtractionResult,
    state: ReviewState,
) -> ReviewState:
    for issue in report.issues:
        current = state.issue_reviews.get(issue.issue_id)
        options = ("not reviewed", "reviewed", "resolved")
        index = options.index(current.status.value) if current else 0
        with st.expander(f"{issue.severity.value.upper()} | {issue.rule_id}: {issue.description}", expanded=False):
            st.json(
                {
                    "issue_id": issue.issue_id,
                    "rule_id": issue.rule_id,
                    "category": issue.category.value,
                    "affected_fields": list(issue.affected_fields),
                    "expected": _short(issue.expected_value),
                    "actual": _short(issue.actual_value),
                    "recommended_action": issue.recommended_action,
                    "resolved_in_detected_report": issue.resolved,
                }
            )
            status_key = _safe_key("issue_status", result.result_id, issue.issue_id)
            note_key = _safe_key("issue_note", result.result_id, issue.issue_id)
            choice = st.selectbox("Review status", options=options, index=index, key=status_key)
            note = st.text_input("Review note", value=current.note or "" if current else "", key=note_key)
            save_col, clear_col = st.columns(2)
            if save_col.button("Save issue review", key=_safe_key("save_issue", result.result_id, issue.issue_id)):
                if choice == "not reviewed":
                    state = remove_issue_review(state, issue.issue_id)
                else:
                    state = set_issue_review(
                        state,
                        issue.issue_id,
                        IssueReviewStatus(choice),
                        note,
                    )
                _get_review_states()[_review_key(result)] = state
                st.rerun()
            if current and clear_col.button("Clear review", key=_safe_key("clear_issue", result.result_id, issue.issue_id)):
                state = remove_issue_review(state, issue.issue_id)
                _get_review_states()[_review_key(result)] = state
                st.rerun()
    return state


def _render_field_review_controls(
    report: QualityReport,
    result: ExtractionResult,
    state: ReviewState,
) -> ReviewState:
    if not report.field_review:
        return state
    rows = [
        {
            "Field": review.field_path,
            "AI value": _short(review.value),
            "Confidence estimate": (
                "not provided" if review.confidence is None else f"{review.confidence:.2f}"
            ),
            "Severity": review.severity.value,
            "Reason": review.reason,
            "Evidence": _short(review.evidence) if review.evidence else "",
        }
        for review in report.field_review
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    for review in report.field_review:
        current = state.field_edits.get(review.field_path)
        default = _display_value(current.edited_value if current else review.value)
        with st.expander(f"Review field: {review.field_path}", expanded=False):
            st.caption(f"Reason: {review.reason}")
            value_key = _safe_key("field_value", result.result_id, review.field_path)
            note_key = _safe_key("field_note", result.result_id, review.field_path)
            edited_text = st.text_input("Reviewed value", value=default, key=value_key)
            note = st.text_input("Review note", value=current.note or "" if current else "", key=note_key)
            save_col, clear_col = st.columns(2)
            if save_col.button("Save field edit", key=_safe_key("save_field", result.result_id, review.field_path)):
                state = set_field_edit(
                    state,
                    review.field_path,
                    _coerce_review_value(review.value, edited_text),
                    note,
                    original_value=review.value,
                )
                _get_review_states()[_review_key(result)] = state
                st.rerun()
            if current and clear_col.button("Clear edit", key=_safe_key("clear_field", result.result_id, review.field_path)):
                state = remove_field_edit(state, review.field_path)
                _get_review_states()[_review_key(result)] = state
                st.rerun()
    return state


def _render_quality_section(
    settings: Settings,
    preprocessing: PreprocessingResult | None,
    result: ExtractionResult,
) -> None:
    st.subheader("7. Extraction quality")
    report = _current_quality_report(settings, preprocessing, result)
    review_states = _get_review_states()
    state = review_states.get(_review_key(result))
    if state is None:
        state = new_review_state(result.result_id)
        review_states[_review_key(result)] = state

    st.caption(
        "Quality status and score are deterministic rule-based indicators, "
        "not AI accuracy, correctness probability, or calibrated confidence. "
        "Confidence values shown by the model are estimates only."
    )
    status_col, score_col, issue_col, review_col = st.columns(4)
    status_col.metric("Quality status", report.quality_status.value)
    score_col.metric("Rule-based indicator", f"{report.quality_score}/100")
    issue_col.metric("Detected issues", report.issue_counts.total)
    review_col.metric("Review required", "yes" if report.review_required else "no")
    st.progress(report.quality_score / 100)
    st.caption(report.assessment_summary)
    st.caption(report.score_explanation)

    with st.expander("Validation summary", expanded=False):
        st.json(report.validation_summary.model_dump(mode="json"))

    st.markdown("#### Detected issues")
    severities = st.multiselect(
        "Severity filter",
        options=[severity.value for severity in QualitySeverity],
        default=[severity.value for severity in QualitySeverity],
        key=_safe_key("severity_filter", result.result_id),
    )
    categories = st.multiselect(
        "Category filter",
        options=[category.value for category in QualityCategory],
        default=[category.value for category in QualityCategory],
        key=_safe_key("category_filter", result.result_id),
    )
    show_resolved = st.checkbox(
        "Show resolved issues",
        value=True,
        key=_safe_key("show_resolved", result.result_id),
    )
    filtered = [
        issue
        for issue in report.issues
        if issue.severity.value in severities
        and issue.category.value in categories
        and (show_resolved or not issue.resolved)
    ]
    if filtered:
        issue_rows = [
            {
                "Issue ID": issue.issue_id,
                "Rule": issue.rule_id,
                "Severity": issue.severity.value,
                "Category": issue.category.value,
                "Description": _short(issue.description),
                "Resolved": "yes" if issue.resolved else "no",
            }
            for issue in filtered
        ]
        st.dataframe(issue_rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No issues match the current filters.")

    st.markdown("#### Review detected issues")
    st.caption("Reviewing an issue records a human action; it does not erase the AI finding or verify the source.")
    state = _render_issue_review_controls(report, result, state)

    st.markdown("#### Review low-confidence fields")
    st.caption("Reviewer values are stored separately. The original AI extraction is never overwritten.")
    state = _render_field_review_controls(report, result, state)

    st.markdown("#### Session review")
    progress = review_progress(report, state)
    st.caption(
        f"Issues reviewed: {progress['reviewed']}/{progress['detected']} | "
        f"resolved: {progress['resolved']}. Review completion does not prove correctness."
    )
    review_note = st.text_area(
        "Add a session review note",
        key=_safe_key("reviewer_note_input", result.result_id),
        max_chars=2000,
    )
    note_col, reset_col = st.columns(2)
    if note_col.button("Add review note", key=_safe_key("add_review_note", result.result_id)):
        state = add_reviewer_note(state, review_note)
        _get_review_states()[_review_key(result)] = state
        st.rerun()
    if reset_col.button("Reset all review changes", key=_safe_key("reset_review", result.result_id)):
        state = reset_review(state)
        _get_review_states()[_review_key(result)] = state
        st.rerun()
    if state.field_edits or state.issue_reviews or state.reviewer_notes:
        st.json(state.model_dump(mode="json"))

    st.markdown("#### Quality and review exports")
    reviewed_report = apply_review_to_report(report, state)
    reviewed_output = build_reviewed_output(result, reviewed_report, state)
    original_json = dump_original_extraction(result)
    quality_json = dump_quality_report(report, state)
    reviewed_json = dump_reviewed_output(reviewed_output)
    markdown_report = render_quality_markdown(report, result, state, reviewed_output)
    tabs = st.tabs(
        [
            "Original extraction JSON",
            "Quality JSON",
            "Reviewed output JSON",
            "Markdown report",
        ]
    )
    exports = (
        ("Download original extraction JSON", original_json, "extraction.json", "original"),
        ("Download quality JSON", quality_json, "quality.json", "quality"),
        ("Download reviewed output JSON", reviewed_json, "reviewed.json", "reviewed"),
        ("Download markdown report", markdown_report, "quality.md", "markdown"),
    )
    for tab, (label, content, file_name, suffix) in zip(tabs, exports):
        with tab:
            st.code(content, language="markdown" if suffix == "markdown" else "json")
            st.download_button(
                label,
                data=content,
                file_name=f"{Path(result.source_document_id or result.result_id).stem}_{file_name}",
                mime="text/markdown" if suffix == "markdown" else "application/json",
                key=_safe_key("download_quality", result.result_id, suffix),
            )
    st.caption(
        "Reviewed output contains session-only edits. It is not verified against "
        "the source unless a reviewer explicitly confirms each field."
    )


def _evaluation_pct(value: object) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.2f}%"


def _run_fixture_evaluation(
    settings: Settings,
    mode: EvaluationMode,
    subset: EvaluationSubset,
) -> tuple[EvaluationRunSummary, dict[str, str]]:
    samples = select_samples(load_dataset(DEFAULT_EVALUATION_ROOT), subset)
    provider = FixtureVisionProvider(
        [sample.ground_truth for sample in samples],
        name="ui_local_fixture" if mode is EvaluationMode.LOCAL_SYNTHETIC_TESTS else "ui_mock_fixture",
    )
    runner = EvaluationRunner(
        settings,
        mode,
        results_root=DEFAULT_EVALUATION_ROOT / "results",
        provider=provider,
        max_workers=1,
        request_delay_seconds=0,
    )
    summary = runner.run(subset=subset, resume=False)
    reports = write_reports(summary, DEFAULT_EVALUATION_ROOT / "reports")
    return summary, reports


def _run_live_evaluation(
    settings: Settings,
    subset: EvaluationSubset,
) -> tuple[EvaluationRunSummary, dict[str, str]]:
    runner = EvaluationRunner(
        settings,
        EvaluationMode.REAL_GEMINI_API_EVALUATION,
        results_root=DEFAULT_EVALUATION_ROOT / "results",
        max_workers=1,
    )
    summary = runner.run(subset=subset, resume=True)
    reports = write_reports(summary, DEFAULT_EVALUATION_ROOT / "reports")
    return summary, reports


def _show_evaluation_summary(summary: EvaluationRunSummary, reports: dict[str, str]) -> None:
    st.success(f"Evaluation {summary.run_id} completed and reports were written.")
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Schema validity",
        _evaluation_pct(summary.schema_validity.get("schema_validity_rate")),
    )
    metric_columns[1].metric(
        "Normalized field accuracy",
        _evaluation_pct(summary.field_metrics.get("normalized_accuracy")),
    )
    metric_columns[2].metric(
        "Document type accuracy",
        _evaluation_pct(summary.document_type_metrics.get("document_type_accuracy")),
    )
    metric_columns[3].metric(
        "API request success",
        _evaluation_pct(summary.performance.get("api_request_success_rate")),
    )
    st.caption(
        f"Mode: {summary.mode.value} | API status: {summary.api_status} | "
        f"Model: {summary.model_id} | Dataset: {summary.dataset_version}"
    )
    with st.expander("Metric details", expanded=False):
        st.json(
            {
                "schema_validity": summary.schema_validity,
                "field_metrics": summary.field_metrics,
                "quality_metrics": summary.quality_metrics,
                "performance": summary.performance,
                "warnings": summary.dataset_validation.get("warnings", []),
            }
        )
    report_text = render_markdown_report(summary)
    st.download_button(
        "Download evaluation report",
        data=report_text,
        file_name=f"{summary.run_id}_report.md",
        mime="text/markdown",
        key="phase5_download_evaluation_report",
    )
    st.json(reports)


def _render_evaluation_section(settings: Settings) -> None:
    st.subheader("8. Evaluation")
    st.caption(
        "Local and mocked evaluations use deterministic fixtures and make no API calls. "
        "Live Gemini evaluation is opt-in, may incur cost and quota use, and is never "
        "started by opening this page."
    )
    tabs = st.tabs(["Dataset", "Local synthetic", "Mocked API", "Live Gemini"])

    with tabs[0]:
        if st.button("Validate evaluation dataset", key="phase5_validate_dataset"):
            try:
                st.json(validate_dataset(DEFAULT_EVALUATION_ROOT))
            except Exception as exc:
                st.error(f"Dataset validation failed: {type(exc).__name__}.")

    for tab, label, mode in (
        (tabs[1], "Run local synthetic tests", EvaluationMode.LOCAL_SYNTHETIC_TESTS),
        (tabs[2], "Run mocked API tests", EvaluationMode.MOCKED_API_TESTS),
    ):
        with tab:
            st.caption(f"Results are labeled **{mode.value}** and never represent a real API model.")
            subset = st.radio(
                f"Subset for {mode.value}",
                [item.value for item in EvaluationSubset],
                index=0,
                horizontal=True,
                key=f"phase5_{mode.value}_subset",
            )
            if st.button(label, key=f"phase5_run_{mode.value}"):
                with st.spinner("Running local evaluation..."):
                    try:
                        summary, reports = _run_fixture_evaluation(
                            settings, mode, EvaluationSubset(subset)
                        )
                        st.session_state[EVALUATION_SUMMARY_KEY] = summary
                        st.session_state[EVALUATION_REPORTS_KEY] = reports
                    except Exception as exc:
                        st.error(f"Evaluation failed: {type(exc).__name__}.")

    with tabs[3]:
        st.write(
            f"API key: **{'configured' if settings.is_api_key_configured else 'not configured'}** | "
            f"Model: `{settings.gemini_model}`"
        )
        st.warning(
            "Live evaluation sends synthetic sample images to Gemini, can incur cost, and "
            "consumes quota. Start with the small subset. Failures are recorded as provider "
            "errors; no results are invented."
        )
        live_subset = st.radio(
            "Live subset",
            [item.value for item in EvaluationSubset],
            index=0,
            horizontal=True,
            key="phase5_live_subset",
        )
        consent = st.checkbox(
            "I understand that this will make real, billable Gemini API requests.",
            key="phase5_live_consent",
        )
        full_consent = True
        if live_subset == EvaluationSubset.FULL.value:
            full_consent = st.checkbox(
                "I confirm the full live subset after completing a small run.",
                key="phase5_live_full_consent",
            )
        live_ready = settings.is_api_key_configured and consent and full_consent
        if st.button("Run live Gemini evaluation", disabled=not live_ready, key="phase5_run_live"):
            with st.spinner("Running live evaluation..."):
                try:
                    summary, reports = _run_live_evaluation(
                        settings, EvaluationSubset(live_subset)
                    )
                    st.session_state[EVALUATION_SUMMARY_KEY] = summary
                    st.session_state[EVALUATION_REPORTS_KEY] = reports
                except Exception as exc:
                    st.error(f"Live evaluation failed: {type(exc).__name__}.")
        if not settings.is_api_key_configured:
            st.info("Set GEMINI_API_KEY in .env and restart the app to enable live evaluation.")

    if EVALUATION_SUMMARY_KEY in st.session_state:
        st.divider()
        _show_evaluation_summary(
            st.session_state[EVALUATION_SUMMARY_KEY],
            st.session_state.get(EVALUATION_REPORTS_KEY, {}),
        )


def _render_environment_diagnostics(settings: Settings) -> None:
    with st.expander("Appendix: environment diagnostics", expanded=False):
        st.caption("Configuration values are display-safe; API keys are never shown.")
        st.json(settings.public_summary())
        module_rows = [
            {
                "Component": label,
                "Status": "available" if _module_is_available(module_name) else "missing",
            }
            for label, module_name in REQUIRED_MODULES.items()
        ]
        dir_rows = [
            {
                "Component": f"directory: {name}",
                "Status": "available" if (PROJECT_ROOT / name).is_dir() else "missing",
            }
            for name in REQUIRED_DIRS
        ]
        st.table(module_rows + dir_rows)
        st.write(
            "Foundation schemas, validators, logging, and configuration remain "
            "available and unchanged in behavior."
        )


def main() -> None:
    st.set_page_config(page_title="Multimodal Document Intelligence", layout="wide")
    configure_logging()
    logger = get_logger("app")

    try:
        settings = get_settings()
    except ConfigurationError as exc:
        detail = sanitize_diagnostic_text(exc) or type(exc).__name__
        st.error(f"Configuration error: {detail}")
        st.stop()
    configure_logging(level=settings.log_level)

    st.title("Multimodal Document Intelligence")
    st.caption(f"Phase 5 - Evaluation & Benchmarking | version {__version__}")
    st.info(
        "Start here: upload documents, process them, review the preview, then extract. "
        "Uploads are validated and preprocessed locally. Prepared page images are sent "
        "to Gemini only after you start extraction and configure GEMINI_API_KEY. "
        "Quality checks are deterministic; review changes are session-only and "
        "never overwrite the original AI extraction. No API key or document "
        "content is displayed in logs."
    )

    session = _get_session()
    session.prune()

    _render_upload_section(session, settings)
    st.divider()
    _render_document_list(session)
    st.divider()
    selected = _render_controls(session, settings)
    if selected is not None:
        result = session.processed_for(selected.document_id)
        if result is not None:
            _render_processing_results(result)
        _render_preview(session, settings, selected)
    st.divider()
    _render_extraction_section(session, settings, selected)
    st.divider()
    _render_evaluation_section(settings)
    st.divider()
    _render_environment_diagnostics(settings)
    logger.info("Phase 5 app rendered with %d document(s).", len(session.documents))


if __name__ == "__main__":
    main()
