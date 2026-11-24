"""Deterministic consistency and arithmetic checks by document type."""

from __future__ import annotations

import re
from collections import Counter
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Mapping

from src.config import Settings
from src.schemas.common import DocumentType, FieldConfidenceEntry
from src.schemas.diagram import DiagramDocument
from src.schemas.handwritten_note import HandwrittenNoteDocument
from src.schemas.invoice import InvoiceDocument
from src.schemas.quality import QualityCategory, QualityIssue, QualitySeverity
from src.schemas.receipt import ReceiptDocument

from .rules import amounts_close, make_issue

_UNCERTAIN_WORDS = ("maybe", "perhaps", "possibly", "might", "unclear", "illegible", "?")
_DIRECT_DATE_RE = re.compile(r"\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b")


def _today(today: date | None) -> date:
    return today or date.today()


def _is_negative(value: Decimal | None) -> bool:
    return value is not None and value < 0


def _currency_values(document: Any) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if ("currency" in key or str(key).endswith("_currency")) and isinstance(child, str):
                    values.append((child_path, child))
                walk(child, child_path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(document.model_dump(mode="python"), "")
    return values


def _currency_issues(document: Any) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    values = _currency_values(document)
    distinct = {value.upper() for _, value in values}
    if len(distinct) > 1:
        issues.append(
            make_issue(
                "DOC-CURR-001",
                QualitySeverity.ERROR,
                QualityCategory.CURRENCY,
                "Currency codes are inconsistent across extracted fields.",
                affected_fields=tuple(path for path, _ in values),
                actual_value=", ".join(sorted(distinct)),
                recommended_action="Compare all printed currency codes on the source document.",
            )
        )
    for path, value in values:
        if not re.fullmatch(r"[A-Z]{3}", value):
            issues.append(
                make_issue(
                    "DOC-CURR-002",
                    QualitySeverity.ERROR,
                    QualityCategory.CURRENCY,
                    "Currency code is not a valid three-letter ISO-style code.",
                    affected_fields=(path,),
                    actual_value=value,
                    recommended_action="Correct the currency code to a three-letter format such as USD or EUR.",
                )
            )
    return issues


def _date_horizon_issues(fields: tuple[str, ...], values: tuple[date | None, ...], settings: Settings, today: date) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    for field, value in zip(fields, values):
        if value is None:
            continue
        if value > today + timedelta(days=settings.future_date_tolerance_days):
            issues.append(
                make_issue(
                    "DOC-DATE-002",
                    QualitySeverity.WARNING,
                    QualityCategory.DATE,
                    f"{field} is unusually far in the future.",
                    affected_fields=(field,),
                    actual_value=value,
                    recommended_action="Verify the date was read correctly from the source.",
                )
            )
        elif value < today - timedelta(days=settings.max_past_date_days):
            issues.append(
                make_issue(
                    "DOC-DATE-003",
                    QualitySeverity.WARNING,
                    QualityCategory.DATE,
                    f"{field} is unusually old.",
                    affected_fields=(field,),
                    actual_value=value,
                    recommended_action="Verify the date was read correctly from the source.",
                )
            )
    return issues


def _invoice_line_issues(document: InvoiceDocument, settings: Settings) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    for index, line in enumerate(document.line_items):
        prefix = f"line_items[{index}]"
        if not line.description or not line.description.strip():
            issues.append(
                make_issue(
                    "INV-LINE-002",
                    QualitySeverity.WARNING,
                    QualityCategory.COMPLETENESS,
                    f"Invoice line {index + 1} has no description.",
                    affected_fields=(f"{prefix}.description",),
                    recommended_action="Check the line description on the source invoice.",
                )
            )
        if _is_negative(line.quantity):
            issues.append(
                make_issue(
                    "INV-LINE-003",
                    QualitySeverity.ERROR,
                    QualityCategory.CONSISTENCY,
                    f"Invoice line {index + 1} has a negative quantity.",
                    affected_fields=(f"{prefix}.quantity",),
                    actual_value=line.quantity,
                    recommended_action="Verify the quantity; negative quantities are not expected here.",
                )
            )
        if _is_negative(line.unit_price):
            issues.append(
                make_issue(
                    "INV-LINE-004",
                    QualitySeverity.ERROR,
                    QualityCategory.CONSISTENCY,
                    f"Invoice line {index + 1} has a negative unit price.",
                    affected_fields=(f"{prefix}.unit_price",),
                    actual_value=line.unit_price,
                    recommended_action="Verify the unit price on the source invoice.",
                )
            )
        if line.amount is None and line.quantity is not None and line.unit_price is not None:
            issues.append(
                make_issue(
                    "INV-LINE-005",
                    QualitySeverity.WARNING,
                    QualityCategory.COMPLETENESS,
                    f"Invoice line {index + 1} amount is missing even though quantity and unit price are present.",
                    affected_fields=(f"{prefix}.amount",),
                    expected_value=line.quantity * line.unit_price,
                    recommended_action="Recheck whether a printed line amount was missed.",
                )
            )
        if line.quantity is None or line.unit_price is None or line.amount is None:
            continue
        expected = line.quantity * line.unit_price
        if not amounts_close(line.amount, expected, settings.amount_tolerance):
            issues.append(
                make_issue(
                    "INV-LINE-001",
                    QualitySeverity.ERROR,
                    QualityCategory.CONSISTENCY,
                    f"Invoice line {index + 1} amount does not match quantity × unit price.",
                    affected_fields=(
                        f"{prefix}.quantity",
                        f"{prefix}.unit_price",
                        f"{prefix}.amount",
                    ),
                    expected_value=expected,
                    actual_value=line.amount,
                    recommended_action="Recheck the line quantity, unit price, and printed amount.",
                )
            )
    return issues


def collect_invoice_issues(
    document: InvoiceDocument,
    settings: Settings,
    today: date | None = None,
) -> list[QualityIssue]:
    """Collect deterministic invoice consistency findings."""
    issues = _invoice_line_issues(document, settings)

    amounts = [line.amount for line in document.line_items if line.amount is not None]
    if document.subtotal is not None and amounts and len(amounts) == len(document.line_items):
        expected_subtotal = sum(amounts, Decimal("0"))
        if not amounts_close(document.subtotal, expected_subtotal, settings.amount_tolerance):
            issues.append(
                make_issue(
                    "INV-TOTAL-001",
                    QualitySeverity.ERROR,
                    QualityCategory.CONSISTENCY,
                    "Invoice subtotal does not match the sum of line item amounts.",
                    affected_fields=("subtotal", "line_items[].amount"),
                    expected_value=expected_subtotal,
                    actual_value=document.subtotal,
                    recommended_action="Compare the printed subtotal with every listed line item amount.",
                )
            )
    elif document.subtotal is not None and amounts:
        issues.append(
            make_issue(
                "INV-TOTAL-004",
                QualitySeverity.INFO,
                QualityCategory.CONSISTENCY,
                "Invoice subtotal could not be fully verified because some line amounts are missing.",
                affected_fields=("subtotal", "line_items[].amount"),
                recommended_action="Review missing line amounts if a full arithmetic check is needed.",
            )
        )

    if document.tax is not None and _is_negative(document.tax):
        issues.append(
            make_issue(
                "INV-TAX-001",
                QualitySeverity.ERROR,
                QualityCategory.CONSISTENCY,
                "Invoice tax is negative.",
                affected_fields=("tax",),
                actual_value=document.tax,
                recommended_action="Verify the printed tax value.",
            )
        )
    if (
        document.tax is not None
        and document.subtotal is not None
        and document.subtotal > 0
        and document.tax > document.subtotal
    ):
        issues.append(
            make_issue(
                "INV-TAX-002",
                QualitySeverity.WARNING,
                QualityCategory.CONSISTENCY,
                "Invoice tax is larger than the subtotal; no tax rate was assumed.",
                affected_fields=("tax", "subtotal"),
                expected_value="tax should not exceed subtotal in typical invoices",
                actual_value={"tax": document.tax, "subtotal": document.subtotal},
                recommended_action="Confirm the tax and subtotal values on the source invoice.",
            )
        )

    if document.total is not None and document.subtotal is not None:
        expected_total = document.subtotal + (document.tax or Decimal("0")) - (
            document.discount or Decimal("0")
        )
        if not amounts_close(document.total, expected_total, settings.amount_tolerance):
            issues.append(
                make_issue(
                    "INV-TOTAL-002",
                    QualitySeverity.ERROR,
                    QualityCategory.CONSISTENCY,
                    "Invoice total is inconsistent with subtotal, tax, and discount.",
                    affected_fields=("subtotal", "tax", "discount", "total"),
                    expected_value=expected_total,
                    actual_value=document.total,
                    recommended_action="Recalculate the total from the printed subtotal, tax, and discount.",
                )
            )
    elif document.total is not None:
        issues.append(
            make_issue(
                "INV-TOTAL-005",
                QualitySeverity.INFO,
                QualityCategory.COMPLETENESS,
                "Invoice total could not be fully verified because the subtotal is missing.",
                affected_fields=("total", "subtotal"),
                recommended_action="Review the printed subtotal if full arithmetic verification is needed.",
            )
        )

    if document.invoice_date and document.due_date and document.due_date < document.invoice_date:
        issues.append(
            make_issue(
                "INV-DATE-001",
                QualitySeverity.ERROR,
                QualityCategory.DATE,
                "Invoice due date is earlier than the invoice date.",
                affected_fields=("invoice_date", "due_date"),
                expected_value=document.invoice_date,
                actual_value=document.due_date,
                recommended_action="Confirm whether the due date was misread.",
            )
        )
    issues.extend(
        _date_horizon_issues(
            ("invoice_date", "due_date"),
            (document.invoice_date, document.due_date),
            settings,
            _today(today),
        )
    )
    issues.extend(_currency_issues(document))
    return issues


def _receipt_item_issues(document: ReceiptDocument, settings: Settings) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    for index, item in enumerate(document.items):
        prefix = f"items[{index}]"
        if not item.description or not item.description.strip():
            issues.append(
                make_issue(
                    "RCPT-LINE-002",
                    QualitySeverity.WARNING,
                    QualityCategory.COMPLETENESS,
                    f"Receipt item {index + 1} has no description.",
                    affected_fields=(f"{prefix}.description",),
                    recommended_action="Check the item description on the source receipt.",
                )
            )
        if item.quantity is None or item.quantity <= 0:
            issues.append(
                make_issue(
                    "RCPT-LINE-003",
                    QualitySeverity.WARNING,
                    QualityCategory.COMPLETENESS,
                    f"Receipt item {index + 1} has a missing or suspicious quantity.",
                    affected_fields=(f"{prefix}.quantity",),
                    actual_value=item.quantity,
                    recommended_action="Verify the printed quantity.",
                )
            )
        if _is_negative(item.unit_price) or _is_negative(item.item_amount):
            issues.append(
                make_issue(
                    "RCPT-LINE-004",
                    QualitySeverity.ERROR,
                    QualityCategory.CONSISTENCY,
                    f"Receipt item {index + 1} contains a negative amount.",
                    affected_fields=(f"{prefix}.unit_price", f"{prefix}.item_amount"),
                    actual_value={"unit_price": item.unit_price, "item_amount": item.item_amount},
                    recommended_action="Verify the printed item amounts.",
                )
            )
        if item.quantity is None or item.unit_price is None or item.item_amount is None:
            continue
        expected = item.quantity * item.unit_price
        if not amounts_close(item.item_amount, expected, settings.amount_tolerance):
            issues.append(
                make_issue(
                    "RCPT-LINE-001",
                    QualitySeverity.ERROR,
                    QualityCategory.CONSISTENCY,
                    f"Receipt item {index + 1} amount does not match quantity × unit price.",
                    affected_fields=(
                        f"{prefix}.quantity",
                        f"{prefix}.unit_price",
                        f"{prefix}.item_amount",
                    ),
                    expected_value=expected,
                    actual_value=item.item_amount,
                    recommended_action="Recheck the item quantity, unit price, and printed amount.",
                )
            )
    return issues


def collect_receipt_issues(
    document: ReceiptDocument,
    settings: Settings,
    today: date | None = None,
) -> list[QualityIssue]:
    """Collect deterministic receipt consistency findings."""
    issues = _receipt_item_issues(document, settings)

    amounts = [item.item_amount for item in document.items if item.item_amount is not None]
    item_sum = sum(amounts, Decimal("0")) if amounts else None
    if (
        document.subtotal is not None
        and item_sum is not None
        and len(amounts) == len(document.items)
        and not amounts_close(document.subtotal, item_sum, settings.amount_tolerance)
    ):
        issues.append(
            make_issue(
                "RCPT-TOTAL-001",
                QualitySeverity.ERROR,
                QualityCategory.CONSISTENCY,
                "Receipt subtotal does not match the sum of item amounts.",
                affected_fields=("subtotal", "items[].item_amount"),
                expected_value=item_sum,
                actual_value=document.subtotal,
                recommended_action="Compare the printed subtotal with every item amount.",
            )
        )

    if document.total is None:
        issues.append(
            make_issue(
                "RCPT-TOTAL-002",
                QualitySeverity.ERROR,
                QualityCategory.COMPLETENESS,
                "Receipt total is missing.",
                affected_fields=("total",),
                recommended_action="Review the printed receipt total.",
            )
        )
    elif document.total == 0:
        issues.append(
            make_issue(
                "RCPT-TOTAL-003",
                QualitySeverity.WARNING,
                QualityCategory.CONSISTENCY,
                "Receipt total is zero, which may indicate a reading issue.",
                affected_fields=("total",),
                actual_value=document.total,
                recommended_action="Verify the printed total.",
            )
        )
    else:
        base = document.subtotal if document.subtotal is not None else item_sum
        if base is not None:
            expected_total = base + (document.tax or Decimal("0"))
            if not amounts_close(document.total, expected_total, settings.amount_tolerance):
                issues.append(
                    make_issue(
                        "RCPT-TOTAL-004",
                        QualitySeverity.ERROR,
                        QualityCategory.CONSISTENCY,
                        "Receipt total is inconsistent with item amounts/subtotal and tax.",
                        affected_fields=("subtotal", "items[].item_amount", "tax", "total"),
                        expected_value=expected_total,
                        actual_value=document.total,
                        recommended_action="Recalculate the total from printed item amounts, tax, and total.",
                    )
                )

    if document.transaction_datetime is None:
        issues.append(
            make_issue(
                "RCPT-DATE-001",
                QualitySeverity.WARNING,
                QualityCategory.COMPLETENESS,
                "Receipt transaction date is missing.",
                affected_fields=("transaction_datetime",),
                recommended_action="Check whether a transaction date is printed on the receipt.",
            )
        )
    else:
        issues.extend(
            _date_horizon_issues(
                ("transaction_datetime",),
                (document.transaction_datetime.date(),),
                settings,
                _today(today),
            )
        )
    issues.extend(_currency_issues(document))
    return issues


def collect_note_issues(
    document: HandwrittenNoteDocument,
    field_confidence: Mapping[str, FieldConfidenceEntry],
    threshold: float,
) -> list[QualityIssue]:
    """Collect deterministic handwritten-note findings."""
    issues: list[QualityIssue] = []
    transcription = document.transcribed_text or ""
    if not transcription.strip():
        issues.append(
            make_issue(
                "NOTE-001",
                QualitySeverity.WARNING,
                QualityCategory.CONTENT,
                "Handwritten note transcription is empty.",
                affected_fields=("transcribed_text",),
                recommended_action="Review the source image and provide a verified transcription.",
            )
        )
    if document.unreadable_markers:
        issues.append(
            make_issue(
                "NOTE-002",
                QualitySeverity.WARNING,
                QualityCategory.CONTENT,
                "Unreadable handwriting markers are present; the transcription is incomplete.",
                affected_fields=("unreadable_markers",),
                actual_value=[marker.marker_id for marker in document.unreadable_markers],
                recommended_action="Review unreadable sections on the source document.",
            )
        )
    if document.language is None:
        issues.append(
            make_issue(
                "NOTE-003",
                QualitySeverity.INFO,
                QualityCategory.COMPLETENESS,
                "Language was not identified confidently and was left unavailable.",
                affected_fields=("language",),
                recommended_action="Set the language only if it can be confirmed from the source.",
            )
        )
    transcription_confidence = field_confidence.get("transcribed_text")
    if transcription_confidence and transcription_confidence.confidence is not None:
        if transcription_confidence.confidence < threshold:
            issues.append(
                make_issue(
                    "NOTE-004",
                    QualitySeverity.WARNING,
                    QualityCategory.CONFIDENCE,
                    "Transcription confidence is below the configured review threshold.",
                    affected_fields=("transcribed_text",),
                    expected_value=f">= {threshold:.2f}",
                    actual_value=transcription_confidence.confidence,
                    recommended_action="Compare the transcription character by character with the source.",
                )
            )
    if any(word in transcription.lower() for word in _UNCERTAIN_WORDS) and _DIRECT_DATE_RE.search(transcription):
        issues.append(
            make_issue(
                "NOTE-005",
                QualitySeverity.WARNING,
                QualityCategory.DATE,
                "A date mention contains uncertainty markers.",
                affected_fields=("transcribed_text", "dates_mentioned"),
                recommended_action="Confirm the date against the source handwriting.",
            )
        )
    for mention in document.dates_mentioned:
        if mention.resolved_date is None:
            issues.append(
                make_issue(
                    "NOTE-006",
                    QualitySeverity.WARNING,
                    QualityCategory.DATE,
                    "A mentioned date could not be resolved confidently.",
                    affected_fields=("dates_mentioned",),
                    actual_value=mention.raw_text,
                    recommended_action="Review the raw date text and resolve it only if visible.",
                )
            )
    for index, action in enumerate(document.action_items):
        if any(word in action.text.lower() for word in _UNCERTAIN_WORDS):
            issues.append(
                make_issue(
                    "NOTE-007",
                    QualitySeverity.WARNING,
                    QualityCategory.CONTENT,
                    f"Action item {index + 1} contains uncertainty markers.",
                    affected_fields=(f"action_items[{index}].text",),
                    actual_value=action.text,
                    recommended_action="Verify the action item before treating it as final.",
                )
            )
    if (document.key_points or document.action_items or document.dates_mentioned) and not transcription.strip():
        issues.append(
            make_issue(
                "NOTE-008",
                QualitySeverity.ERROR,
                QualityCategory.COMPLETENESS,
                "AI-generated key points or interpretations are present without a direct transcription for support.",
                affected_fields=("transcribed_text", "key_points", "action_items", "dates_mentioned"),
                recommended_action="Review the source handwriting before relying on these interpretations.",
            )
        )
    if document.key_points:
        issues.append(
            make_issue(
                "NOTE-009",
                QualitySeverity.INFO,
                QualityCategory.REVIEW,
                "Key points are AI-generated interpretations, not a direct transcription.",
                affected_fields=("key_points",),
                recommended_action="Treat key points as summaries and verify them against the source if needed.",
            )
        )
    return issues


def collect_diagram_issues(document: DiagramDocument) -> list[QualityIssue]:
    """Collect deterministic diagram findings."""
    issues: list[QualityIssue] = []
    if not document.title:
        issues.append(
            make_issue(
                "DIAG-001",
                QualitySeverity.WARNING,
                QualityCategory.COMPLETENESS,
                "Diagram title is missing.",
                affected_fields=("title",),
                recommended_action="Confirm whether the diagram has a visible title.",
            )
        )
    if not document.elements:
        issues.append(
            make_issue(
                "DIAG-002",
                QualitySeverity.ERROR,
                QualityCategory.COMPLETENESS,
                "Diagram has no extracted elements.",
                affected_fields=("elements",),
                recommended_action="Review whether elements were missed or the image was unclear.",
            )
        )
    element_ids = [element.element_id for element in document.elements]
    duplicates = [element_id for element_id, count in Counter(element_ids).items() if count > 1]
    if duplicates:
        issues.append(
            make_issue(
                "DIAG-003",
                QualitySeverity.ERROR,
                QualityCategory.CONSISTENCY,
                "Diagram contains duplicate element IDs.",
                affected_fields=("elements[].element_id",),
                actual_value=duplicates,
                recommended_action="Correct duplicate identifiers before using this diagram data.",
            )
        )
    known_ids = set(element_ids)
    for index, connection in enumerate(document.connections):
        invalid = [
            value
            for value in (connection.source_id, connection.target_id)
            if value not in known_ids
        ]
        if invalid:
            issues.append(
                make_issue(
                    "DIAG-004",
                    QualitySeverity.ERROR,
                    QualityCategory.CONSISTENCY,
                    f"Diagram connection {index + 1} references an unknown element.",
                    affected_fields=(
                        f"connections[{index}].source_id",
                        f"connections[{index}].target_id",
                    ),
                    actual_value=invalid,
                    recommended_action="Verify the connection endpoints against visible elements.",
                )
            )
        if connection.label is None and connection.relationship_type is None:
            issues.append(
                make_issue(
                    "DIAG-005",
                    QualitySeverity.WARNING,
                    QualityCategory.COMPLETENESS,
                    f"Diagram connection {index + 1} has no label or relationship type.",
                    affected_fields=(
                        f"connections[{index}].label",
                        f"connections[{index}].relationship_type",
                    ),
                    recommended_action="Review the connection label on the source diagram.",
                )
            )
        if "?" in (connection.label or "") or "unclear" in (connection.label or "").lower():
            issues.append(
                make_issue(
                    "DIAG-006",
                    QualitySeverity.WARNING,
                    QualityCategory.REVIEW,
                    f"Diagram connection {index + 1} direction is ambiguous.",
                    affected_fields=(f"connections[{index}].label",),
                    actual_value=connection.label,
                    recommended_action="Check the arrow direction on the original diagram.",
                )
            )
    for index, element in enumerate(document.elements):
        if element.description == "":
            issues.append(
                make_issue(
                    "DIAG-007",
                    QualitySeverity.INFO,
                    QualityCategory.CONTENT,
                    f"Diagram element {index + 1} has an empty description.",
                    affected_fields=(f"elements[{index}].description",),
                    recommended_action="Review whether the description was omitted.",
                )
            )
        if element.label is None:
            issues.append(
                make_issue(
                    "DIAG-008",
                    QualitySeverity.WARNING,
                    QualityCategory.COMPLETENESS,
                    f"Diagram element {index + 1} has no label.",
                    affected_fields=(f"elements[{index}].label",),
                    recommended_action="Check whether the element label is visible in the source.",
                )
            )
    pairs = [(connection.source_id, connection.target_id) for connection in document.connections]
    for (source, target), count in Counter(pairs).items():
        if count > 1:
            issues.append(
                make_issue(
                    "DIAG-009",
                    QualitySeverity.WARNING,
                    QualityCategory.CONSISTENCY,
                    "Diagram contains duplicate connections with potentially conflicting relationships.",
                    affected_fields=("connections",),
                    actual_value={"source_id": source, "target_id": target, "count": count},
                    recommended_action="Compare the duplicate connections on the source diagram.",
                )
            )
    if document.ambiguities:
        issues.append(
            make_issue(
                "DIAG-010",
                QualitySeverity.WARNING,
                QualityCategory.REVIEW,
                "The extraction recorded visual ambiguities that need human review.",
                affected_fields=("ambiguities",),
                actual_value=[ambiguity.description for ambiguity in document.ambiguities],
                recommended_action="Review ambiguous regions in the original diagram.",
            )
        )
    return issues
