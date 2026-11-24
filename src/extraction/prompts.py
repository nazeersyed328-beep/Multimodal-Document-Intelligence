"""Prompt templates for Gemini vision extraction.

Prompts instruct the model to return only schema-constrained JSON, to report
only visible evidence, and to mark unreadable or uncertain information instead
of guessing.
"""

from __future__ import annotations

from ..schemas.common import DocumentType

_METADATA_RULE = """Do not include document_id, source_filename, extracted_at,
validation_status, errors, or warnings inside `data`; the application supplies
those values. Put only document-specific extracted fields in `data`."""

SYSTEM_PROMPT = """You extract structured data from document page images.

Rules:
- Return only JSON that conforms to the schema supplied by the caller.
- Report only information that is visibly present in the supplied images.
- Never guess, infer from location or branding alone, or invent a value.
- Use null for any field that is absent, uncertain, or not readable.
- List absent field names in `missing_fields` and unreadable field names in
  `unreadable_fields`.
- For handwriting, represent illegible spans with explicit unreadable markers
  instead of a guessed transcription.
- For diagrams, record ambiguities when the visual evidence is unclear.
- Monetary values must be numeric strings with at most two decimal places;
  never use floating-point notation or scientific notation.
- Dates must use ISO 8601 (YYYY-MM-DD) and datetimes ISO 8601 with time.
- Currency codes must be three uppercase letters, for example USD.
- Confidence values are model estimates between 0 and 1. Use null when you
  cannot make a responsible estimate. They are not calibrated probabilities.
- Include short evidence snippets when visible text supports a value.
- When multiple page images are supplied, preserve page order and use the page
  labels in the prompt to keep page/document association clear.
"""

DOCUMENT_INSTRUCTIONS: dict[DocumentType, str] = {
    DocumentType.INVOICE: """Extract invoice fields: vendor name and address,
invoice number, invoice date, due date, currency, subtotal, tax, discount,
total, customer details, and line items with description, quantity, unit price,
and amount. Do not compute values that are not printed on the invoice.""",
    DocumentType.RECEIPT: """Extract receipt fields: merchant name, store
address, receipt number if printed, transaction date and time, currency,
purchased items with quantity, unit price, and item amount, subtotal if printed,
tax, total, and payment method if visible. Do not infer a merchant or payment method from
branding alone when no explicit text is present.""",
    DocumentType.HANDWRITTEN_NOTE: """Transcribe visible handwriting faithfully.
Set the language only when you can identify it confidently. Extract key points,
action items, and mentioned dates. For every span you cannot read, add an
unreadable marker with a unique ID and reference that ID in the transcription.
Never silently guess an illegible word. Keep direct transcription separate from
any summary or key points.""",
    DocumentType.DIAGRAM: """Describe the diagram using only visible evidence.
Identify the diagram type only when it is clear. List elements with stable IDs,
connections that reference those IDs, visible labels, and extracted text.
Write a short description and record every ambiguity in the ambiguities list.
Do not interpret the meaning of an unlabeled shape. Mark direction as uncertain
in an ambiguity when an arrowhead or relationship direction is unclear.""",
}


def build_detection_prompt(page_count: int) -> str:
    """Ask the model to classify the document without extracting its contents."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"You are classifying a document with {page_count} supplied page image(s).\n"
        "Choose exactly one of: invoice, receipt, handwritten_note, diagram, or "
        "unknown. Choose unknown when the type is not clear. Do not extract "
        "document fields in this step.\n"
        + _METADATA_RULE
    )


def build_extraction_prompt(document_type: DocumentType, page_count: int) -> str:
    """Combine system rules with instructions for one supported document type."""
    if document_type not in DOCUMENT_INSTRUCTIONS:
        raise ValueError(f"No prompt template defined for {document_type!r}.")
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"DOCUMENT TYPE: {document_type.value}\n"
        f"PAGE IMAGES SUPPLIED: {page_count}\n\n"
        f"{DOCUMENT_INSTRUCTIONS[document_type]}\n\n"
        + _METADATA_RULE
    )


def build_correction_prompt(
    document_type: DocumentType,
    validation_messages: list[str],
) -> str:
    """Ask the model to correct a bounded set of structural validation errors."""
    issues = "\n".join(f"- {message}" for message in validation_messages[:20])
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"DOCUMENT TYPE: {document_type.value}\n\n"
        "Your previous response did not satisfy the required JSON schema. "
        "Return a corrected JSON object that follows the schema exactly. "
        "Keep all previously correct visible evidence, and do not invent new "
        "values to satisfy the schema.\n\n"
        f"Validation issues:\n{issues}\n\n"
        + _METADATA_RULE
    )
