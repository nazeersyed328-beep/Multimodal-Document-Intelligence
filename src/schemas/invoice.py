"""Strict schema for invoice documents."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field, model_validator

from .common import (
    CurrencyCode,
    DocumentBase,
    DocumentType,
    MonetaryAmount,
    QuantityValue,
    StrictModel,
)


class CustomerInfo(StrictModel):
    """Buyer or customer details when they are visible on the invoice."""

    name: str | None = Field(default=None, max_length=300)
    address: str | None = Field(default=None, max_length=1000)
    tax_id: str | None = Field(default=None, max_length=100)
    email: str | None = Field(default=None, max_length=300)


class InvoiceLineItem(StrictModel):
    """One invoice line. Every value stays optional when not visible."""

    description: str = Field(min_length=1, max_length=1000)
    quantity: QuantityValue | None = None
    unit_price: MonetaryAmount | None = None
    amount: MonetaryAmount | None = None


class InvoiceDocument(DocumentBase):
    """Typed payload for an invoice."""

    document_type: Literal[DocumentType.INVOICE] = DocumentType.INVOICE
    vendor_name: str | None = Field(default=None, max_length=300)
    vendor_address: str | None = Field(default=None, max_length=1000)
    invoice_number: str | None = Field(default=None, max_length=100)
    invoice_date: date | None = None
    due_date: date | None = None
    currency: CurrencyCode | None = None
    subtotal: MonetaryAmount | None = None
    tax: MonetaryAmount | None = None
    discount: MonetaryAmount | None = None
    total: MonetaryAmount | None = None
    customer: CustomerInfo | None = None
    line_items: list[InvoiceLineItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_date_order(self) -> InvoiceDocument:
        if self.invoice_date and self.due_date and self.due_date < self.invoice_date:
            raise ValueError("due_date cannot be earlier than invoice_date.")
        return self
