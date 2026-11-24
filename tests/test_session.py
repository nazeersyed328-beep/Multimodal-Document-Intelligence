"""Tests for session-scoped document state helpers."""

from __future__ import annotations

from src.document.ingestion import ingest_upload
from src.document.session import DocumentSession
from src.schemas.document_ingestion import PreprocessingResult, ProcessingStatus

from .conftest import FakeUpload, make_png, make_test_settings


def make_record(name: str, color: tuple[int, int, int], upload_id: str | None = None):
    settings = make_test_settings()
    _, record = ingest_upload(FakeUpload(name, make_png(color=color), upload_id), settings)
    assert record is not None
    return record


def test_add_and_duplicate_detection() -> None:
    session = DocumentSession()
    first = make_record("first.png", (1, 2, 3), "upload-1")
    duplicate = make_record("copy.png", (1, 2, 3), "upload-2")
    assert session.add_record(first) is True
    assert session.add_record(duplicate) is False
    assert len(session.documents) == 1
    assert "upload-1" in session.seen_upload_ids
    assert "upload-2" not in session.seen_upload_ids


def test_remove_also_removes_processed_result() -> None:
    session = DocumentSession()
    record = make_record("page.png", (5, 5, 5), "upload-1")
    session.add_record(record)
    result = PreprocessingResult(
        document_id=record.document_id,
        original_filename=record.sanitized_filename,
        document_format=record.document_format,
        status=ProcessingStatus.COMPLETED,
        processing_time_seconds=0.01,
    )
    assert session.set_processed(record.document_id, result) is True
    assert session.processed_for(record.document_id) is result
    assert session.remove(record.document_id) is True
    assert session.processed_for(record.document_id) is None
    assert "upload-1" not in session.seen_upload_ids


def test_set_processed_requires_existing_document() -> None:
    session = DocumentSession()
    result = PreprocessingResult(
        document_id="doc-missing",
        original_filename="missing.png",
        document_format="png",
        status=ProcessingStatus.COMPLETED,
        processing_time_seconds=0.01,
    )
    assert session.set_processed("doc-missing", result) is False
    assert session.processed == {}


def test_prune_removes_stale_results() -> None:
    session = DocumentSession()
    record = make_record("page.png", (9, 9, 9))
    session.add_record(record)
    session.processed["doc-stale"] = PreprocessingResult(
        document_id="doc-stale",
        original_filename="stale.png",
        document_format="png",
        status=ProcessingStatus.COMPLETED,
        processing_time_seconds=0.0,
    )
    assert session.prune() == 1
    assert "doc-stale" not in session.processed


def test_enforce_limit_removes_oldest_first() -> None:
    session = DocumentSession()
    records = [make_record(f"p{i}.png", (i, 1, 2), str(i)) for i in range(4)]
    for record in records:
        session.add_record(record)
    removed = session.enforce_limit(2)
    assert len(removed) == 2
    assert removed[0] == records[0].document_id
    assert list(session.documents) == [
        records[2].document_id,
        records[3].document_id,
    ]


def test_clear_resets_everything() -> None:
    session = DocumentSession()
    session.add_record(make_record("page.png", (1, 1, 1), "u1"))
    session.mark_upload_seen("u2")
    session.rejected.append("issue")
    session.clear()
    assert session.documents == {}
    assert session.processed == {}
    assert session.rejected == []
    assert session.seen_upload_ids == set()


def test_upload_seen_helpers() -> None:
    session = DocumentSession()
    assert session.is_upload_seen(None) is False
    session.mark_upload_seen("abc")
    assert session.is_upload_seen("abc") is True
    session.mark_upload_seen(None)
    assert session.seen_upload_ids == {"abc"}


def test_total_bytes_reporting() -> None:
    session = DocumentSession()
    record = make_record("page.png", (3, 3, 3))
    session.add_record(record)
    assert session.total_raw_bytes() == record.size_bytes
    assert session.total_processed_bytes() == 0
