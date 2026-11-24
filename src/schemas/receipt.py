"""Strict schema for receipt documents."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from .common import (
    CurrencyCode,
    DocumentBase,
    DocumentType,
    MonetaryAmount,
    QuantityValue,
    StrictModel,
)


class ReceiptItem(StrictModel):
    """One purchased item on a receipt."""

    description: str = Field(min_length=1, max_length=1000)
    quantity: QuantityValue | None = None
    unit_price: MonetaryAmount | None = None
    item_amount: MonetaryAmount | None = None


class ReceiptDocument(DocumentBase):
    """Typed payload for a receipt."""

    document_type: Literal[DocumentType.RECEIPT] = DocumentType.RECEIPT
    merchant_name: str | None = Field(default=None, max_length=300)
    store_address: str | None = Field(default=None, max_length=1000)
    receipt_number: str | None = Field(default=None, max_length=100)
    transaction_datetime: datetime | None = None
    currency: CurrencyCode | None = None
    items: list[ReceiptItem] = Field(default_factory=list)
    subtotal: MonetaryAmount | None = None
    tax: MonetaryAmount | None = None
    total: MonetaryAmount | None = None
    payment_method: str | None = Field(default=None, max_length=100)
