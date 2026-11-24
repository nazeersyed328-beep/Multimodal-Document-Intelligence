"""Session-scoped helpers for uploaded documents and processed results.

The container is plain Python so it can be tested without Streamlit. The app
stores one instance in ``st.session_state`` across reruns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas.document_ingestion import IngestionResult, PreprocessingResult
from .ingestion import DocumentRecord


@dataclass
class DocumentSession:
    """Mutable but bounded collection of documents for one browser session."""

    documents: dict[str, DocumentRecord] = field(default_factory=dict)
    rejected: list[IngestionResult] = field(default_factory=list)
    processed: dict[str, PreprocessingResult] = field(default_factory=dict)
    seen_upload_ids: set[str] = field(default_factory=set)

    @property
    def fingerprints(self) -> set[str]:
        return {record.fingerprint for record in self.documents.values()}

    def mark_upload_seen(self, upload_id: str | None) -> None:
        if upload_id:
            self.seen_upload_ids.add(upload_id)

    def is_upload_seen(self, upload_id: str | None) -> bool:
        return bool(upload_id) and upload_id in self.seen_upload_ids

    def add_record(self, record: DocumentRecord) -> bool:
        """Add a record unless its content fingerprint is already present."""
        if record.fingerprint in self.fingerprints:
            return False
        self.documents[record.document_id] = record
        self.mark_upload_seen(record.upload_id)
        return True

    def add_rejected(self, results: list[IngestionResult] | tuple[IngestionResult, ...]) -> None:
        self.rejected.extend(results)

    def get(self, document_id: str) -> DocumentRecord | None:
        return self.documents.get(document_id)

    def remove(self, document_id: str) -> bool:
        """Remove one document together with its processed payload."""
        record = self.documents.pop(document_id, None)
        self.processed.pop(document_id, None)
        if record is not None and record.upload_id:
            self.seen_upload_ids.discard(record.upload_id)
        return record is not None

    def set_processed(self, document_id: str, result: PreprocessingResult) -> bool:
        """Store processed pages only for a document that still exists."""
        if document_id not in self.documents:
            return False
        self.processed[document_id] = result
        return True

    def processed_for(self, document_id: str) -> PreprocessingResult | None:
        return self.processed.get(document_id)

    def prune(self) -> int:
        """Drop stale processed results whose documents were removed."""
        stale = set(self.processed) - set(self.documents)
        for document_id in stale:
            del self.processed[document_id]
        return len(stale)

    def enforce_limit(self, max_documents: int) -> list[str]:
        """Drop the oldest documents when the configured limit is exceeded."""
        removed: list[str] = []
        while len(self.documents) > max_documents:
            document_id = next(iter(self.documents))
            self.remove(document_id)
            removed.append(document_id)
        return removed

    def clear(self) -> None:
        self.documents.clear()
        self.rejected.clear()
        self.processed.clear()
        self.seen_upload_ids.clear()

    def total_raw_bytes(self) -> int:
        return sum(record.size_bytes for record in self.documents.values())

    def total_processed_bytes(self) -> int:
        return sum(
            page.processed_width * page.processed_height * 3
            for result in self.processed.values()
            for page in result.pages
        )
