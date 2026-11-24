"""Generate deterministic synthetic evaluation documents."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

EVALUATION_ROOT = Path(__file__).resolve().parent
SAMPLES_ROOT = EVALUATION_ROOT / "samples"

WIDTH = 1200
HEIGHT = 800
BACKGROUND = "white"
FOREGROUND = "black"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "C:\\Windows\\Fonts\\consola.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    return image, ImageDraw.Draw(image)


def _lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    *,
    x: int = 60,
    y: int = 50,
    size: int = 24,
    spacing: int = 12,
) -> None:
    font = _font(size)
    current_y = y
    for line in lines:
        draw.text((x, current_y), line, fill=FOREGROUND, font=font)
        current_y += size + spacing


def _save_png(image: Image.Image, name: str) -> Path:
    path = SAMPLES_ROOT / name
    image.save(path, format="PNG", optimize=False)
    return path


def _invoice_clean() -> Path:
    image, draw = _canvas()
    _lines(
        draw,
        [
            "INVOICE",
            "Vendor: Acme Test Supply",
            "Invoice Number: INV-1001",
            "Invoice Date: 2026-01-02",
            "Due Date: 2026-01-20",
            "Currency: USD",
            "",
            "Item                Qty   Unit Price   Amount",
            "Widget A             2       25.00      50.00",
            "Widget B             1       50.00      50.00",
            "",
            "Subtotal: 100.00",
            "Tax: 10.00",
            "Total: 110.00",
        ],
        size=26,
        spacing=10,
    )
    return _save_png(image, "inv-clean-001.png")


def _invoice_missing() -> Path:
    image, draw = _canvas()
    _lines(
        draw,
        [
            "INVOICE",
            "Vendor: Northwind Demo Co",
            "Invoice Number: ________",
            "Invoice Date: 2026-02-05",
            "Due Date: not printed",
            "Currency: USD",
            "",
            "Item                Qty   Unit Price   Amount",
            "Support Session       1       40.00   ________",
            "",
            "Subtotal: 40.00",
            "Tax: not printed",
            "Total: not printed",
        ],
        size=26,
        spacing=10,
    )
    return _save_png(image, "inv-missing-001.png")


def _invoice_arithmetic() -> Path:
    image, draw = _canvas()
    _lines(
        draw,
        [
            "INVOICE",
            "Vendor: Delta Test Goods",
            "Invoice Number: INV-1003",
            "Invoice Date: 2026-03-10",
            "Currency: USD",
            "",
            "Item                Qty   Unit Price   Amount",
            "Widget C             3       20.00      60.00",
            "",
            "Subtotal: 60.00",
            "Tax: 6.00",
            "Total: 99.00",
            "",
            "Note: printed total intentionally does not match subtotal plus tax.",
        ],
        size=26,
        spacing=10,
    )
    return _save_png(image, "inv-arith-001.png")


def _invoice_multipage() -> Path:
    first, first_draw = _canvas()
    _lines(
        first_draw,
        [
            "INVOICE - PAGE 1 OF 2",
            "Vendor: Summit Synthetic Ltd",
            "Invoice Number: INV-1004",
            "Invoice Date: 2026-04-01",
            "Currency: USD",
            "",
            "Subtotal: 120.00",
            "Tax: 12.00",
            "Total: 132.00",
            "",
            "Line items continue on page 2.",
        ],
        size=28,
        spacing=12,
    )
    second, second_draw = _canvas()
    _lines(
        second_draw,
        [
            "INVOICE - PAGE 2 OF 2",
            "Line items continued",
            "",
            "Item                   Qty   Unit Price   Amount",
            "Assembly Service         2       40.00      80.00",
            "Inspection Service       2       20.00      40.00",
        ],
        size=26,
        spacing=12,
    )
    path = SAMPLES_ROOT / "inv-multipage-001.pdf"
    first.save(path, format="PDF", save_all=True, append_images=[second], resolution=100.0)
    return path


def _receipt_clean() -> Path:
    image, draw = _canvas()
    _lines(
        draw,
        [
            "RECEIPT",
            "Merchant: Sample Mart",
            "Date: 2026-01-05 10:30",
            "Currency: USD",
            "",
            "Milk     2 x 3.00    6.00",
            "Bread    1 x 4.00    4.00",
            "",
            "Subtotal: 10.00",
            "Tax: 0.80",
            "Total: 10.80",
        ],
        size=28,
        spacing=12,
    )
    return _save_png(image, "rcpt-clean-001.png")


def _receipt_blurry() -> Path:
    image, draw = _canvas()
    _lines(
        draw,
        [
            "RECEIPT",
            "Merchant: Quick Cafe",
            "Date: 2026-01-06 08:15",
            "Currency: USD",
            "",
            "Coffee      2 x 2.50    5.00",
            "Sandwich    1 x 6.00    6.00",
            "",
            "Subtotal: 11.00",
            "Tax: 1.10",
            "Total: 12.10",
        ],
        size=26,
        spacing=12,
    )
    image = image.filter(ImageFilter.GaussianBlur(radius=1.8))
    return _save_png(image, "rcpt-blurry-001.png")


def _receipt_future() -> Path:
    image, draw = _canvas()
    _lines(
        draw,
        [
            "RECEIPT",
            "Merchant: Future Shop",
            "Date: 2099-01-01 12:00",
            "Currency: USD",
            "",
            "Notebook     1 x 5.00    5.00",
            "",
            "Subtotal: 5.00",
            "Tax: 0.00",
            "Total: 5.00",
        ],
        size=28,
        spacing=12,
    )
    return _save_png(image, "rcpt-future-001.png")


def _note_clean() -> Path:
    image, draw = _canvas()
    _lines(
        draw,
        [
            "HANDWRITTEN NOTE",
            "Title: Team Sync",
            "Language: en",
            "",
            "Call Sam at 10 AM on 2026-01-08.",
            "Bring the budget draft.",
        ],
        size=30,
        spacing=16,
    )
    return _save_png(image, "note-clean-001.png")


def _note_unreadable() -> Path:
    image, draw = _canvas()
    _lines(
        draw,
        [
            "HANDWRITTEN NOTE",
            "Title: Lab Notes",
            "",
            "[ILLEGIBLE] was reviewed by [ILLEGIBLE]",
            "on 2026-0?-??.",
            "Follow up next week.",
        ],
        size=30,
        spacing=16,
    )
    return _save_png(image, "note-unreadable-001.png")


def _box(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], label: str) -> None:
    draw.rectangle(rect, outline=FOREGROUND, width=3)
    draw.text((rect[0] + 18, rect[1] + 18), label, fill=FOREGROUND, font=_font(24))


def _diagram_clean() -> Path:
    image, draw = _canvas()
    _lines(draw, ["FLOWCHART: Order Flow"], size=32, spacing=16)
    _box(draw, (80, 250, 300, 360), "Start")
    _box(draw, (470, 250, 700, 360), "Review")
    _box(draw, (870, 250, 1100, 360), "Done")
    draw.line((300, 305, 470, 305), fill=FOREGROUND, width=3)
    draw.text((335, 265), "next", fill=FOREGROUND, font=_font(22))
    draw.line((700, 305, 870, 305), fill=FOREGROUND, width=3)
    draw.text((735, 265), "approved", fill=FOREGROUND, font=_font(22))
    return _save_png(image, "diag-clean-001.png")


def _diagram_ambiguous() -> Path:
    image, draw = _canvas()
    _lines(draw, ["DIAGRAM: Approval Path"], size=32, spacing=16)
    _box(draw, (80, 240, 330, 360), "Request")
    _box(draw, (500, 240, 730, 360), "Manager")
    _box(draw, (900, 240, 1120, 360), "Archive")
    draw.line((330, 300, 500, 300), fill=FOREGROUND, width=3)
    draw.text((385, 260), "?", fill=FOREGROUND, font=_font(28))
    draw.line((730, 300, 900, 300), fill=FOREGROUND, width=3)
    draw.text((760, 260), "unclear", fill=FOREGROUND, font=_font(22))
    _lines(
        draw,
        ["Ambiguity: arrow direction between Request and Manager is unclear."],
        x=60,
        y=520,
        size=24,
        spacing=10,
    )
    return _save_png(image, "diag-ambiguous-001.png")


def generate() -> list[Path]:
    SAMPLES_ROOT.mkdir(parents=True, exist_ok=True)
    paths = [
        _invoice_clean(),
        _invoice_missing(),
        _invoice_arithmetic(),
        _invoice_multipage(),
        _receipt_clean(),
        _receipt_blurry(),
        _receipt_future(),
        _note_clean(),
        _note_unreadable(),
        _diagram_clean(),
        _diagram_ambiguous(),
    ]
    manifest = {
        "generator": "evaluation/generate_samples.py",
        "dataset_version": "phase5-synthetic-v1",
        "license": "CC0-1.0 for these synthetic assets",
        "files": [
            {
                "file_name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in paths
        ],
    }
    (SAMPLES_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return paths


if __name__ == "__main__":
    generated = generate()
    print(f"Generated {len(generated)} synthetic evaluation samples.")
