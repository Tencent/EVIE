#!/usr/bin/env python3
"""Write the bundled 8-page retrieval demo (synthetic pages, not a benchmark)."""

from __future__ import annotations

import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
PAGES_DIR = ROOT / "pages"
PARQUET_DIR = ROOT / "demo" / "evie_pages" / "data"

FONT_CANDIDATES = (
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)

PAGES = [
    {
        "file": "q3_revenue.png",
        "bg": (24, 72, 140),
        "title": "FY2024 Q3 earnings",
        "body": "Q3 2024 revenue\n$12.4 million\n\nOperating margin 18%",
        "query": "What was Q3 2024 revenue?",
    },
    {
        "file": "org_chart.png",
        "bg": (20, 110, 64),
        "title": "Org chart",
        "body": "CEO: Dana Ng\n  VP Sales: Priya Shah\n  VP Eng: Ken Ortiz",
        "query": "Who is the VP of Sales under the CEO?",
    },
    {
        "file": "invoice_4412.png",
        "bg": (150, 70, 24),
        "title": "INVOICE 4412",
        "body": "DUE DATE\n14 March 2026\nPayee: North Harbor Co.\nAmount due: $8,150",
        "query": "What is the due date on invoice 4412?",
    },
    {
        "file": "warehouse_b.png",
        "bg": (16, 96, 118),
        "title": "WAREHOUSE B",
        "body": "LOCATION: NORTH LOT\nGate 4\nForklifts: 12",
        "query": "Where is Warehouse B located?",
    },
    {
        "file": "traceback.png",
        "bg": (120, 28, 36),
        "title": "CI failure",
        "body": "TypeError: NoneType\nFile app/ledger.py\nline 88, in settle()",
        "query": "Which file and line raised the TypeError?",
    },
    {
        "file": "cash.png",
        "bg": (48, 48, 48),
        "title": "Treasury snapshot",
        "body": "Cash on hand\n$80 million\nT-bills: $12 million",
        "query": "How much cash is on hand?",
    },
    {
        "file": "headcount.png",
        "bg": (72, 40, 120),
        "title": "People ops",
        "body": "Employee headcount\n1,204\nOpen reqs: 17",
        "query": "What is the employee headcount?",
    },
    {
        "file": "patent.png",
        "bg": (40, 70, 40),
        "title": "IP docket",
        "body": "Patent US-11,234,567\nTitle: Token pooling\nGranted 2025-11-02",
        "query": "What is the granted patent number?",
    },
]


def _font(size: int) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _png(bg: tuple[int, int, int], title: str, body: str) -> bytes:
    im = Image.new("RGB", (768, 1024), bg)
    draw = ImageDraw.Draw(im)
    draw.rectangle((28, 28, 740, 996), outline=(255, 255, 255), width=6)
    draw.text((56, 64), title, fill=(255, 255, 230), font=_font(42))
    draw.text((56, 180), body, fill=(255, 255, 255), font=_font(36), spacing=10)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    pngs = []
    for page in PAGES:
        blob = _png(page["bg"], page["title"], page["body"])
        (PAGES_DIR / page["file"]).write_bytes(blob)
        pngs.append(blob)

    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    table = pa.table(
        {
            "query": [p["query"] for p in PAGES],
            "image_filename": [p["file"] for p in PAGES],
            "image": pa.array(
                [{"bytes": b, "path": p["file"]} for b, p in zip(pngs, PAGES)],
                type=image_type,
            ),
        }
    )
    out = PARQUET_DIR / "test-00000-of-00001.parquet"
    pq.write_table(table, out)
    print(f"wrote {len(PAGES)} pages -> {PAGES_DIR}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
