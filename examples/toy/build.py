#!/usr/bin/env python3
"""Write the bundled toy pages + hard-negative index (synthetic, not a real dump)."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
PAGES = [
    (0, (30, 90, 180), "Q3 revenue\n$12.4M"),
    (1, (20, 140, 70), "Cash on hand\n$80M"),
    (2, (200, 110, 30), "Org chart\nCEO > VP Sales"),
    (3, (120, 50, 160), "Invoice 4412\ndue in 30 days"),
    (4, (20, 130, 140), "Warehouse B\nnorth lot"),
    (5, (160, 40, 40), "TypeError\nline 88"),
]
QUERIES = [
    ("What was Q3 revenue?", [0], [1, 5]),
    ("Who reports to the CEO?", [2], [0, 3]),
    ("When is invoice 4412 due?", [3], [1, 4]),
    ("Which warehouse is on the map?", [4], [2, 5]),
    ("Any financial figures on the page?", [0, 1], [2, 5]),
]


def _png(rgb: tuple[int, int, int], caption: str) -> bytes:
    im = Image.new("RGB", (256, 256), rgb)
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    draw.rectangle((12, 12, 244, 244), outline=(255, 255, 255), width=3)
    draw.multiline_text((24, 96), caption, fill=(255, 255, 255), font=font, spacing=6)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    pngs = [_png(color, text) for _, color, text in PAGES]
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    table = pa.table(
        {
            "doc_id": [p[0] for p in PAGES],
            "query": [f"page {p[0]}" for p in PAGES],
            "query_normalized": [f"page {p[0]}" for p in PAGES],
            "source": ["colpali_train_set"] * len(PAGES),
            "language": ["en"] * len(PAGES),
            "image_sha256": [hashlib.sha256(b).hexdigest() for b in pngs],
            "image": pa.array(
                [{"bytes": b, "path": f"page_{i}.png"} for i, b in enumerate(pngs)],
                type=image_type,
            ),
        }
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, ROOT / "part_000.parquet")

    judged = {
        "query": [q for q, _, _ in QUERIES],
        "positive_doc_id": [pos[0] for _, pos, _ in QUERIES],
        "positive_doc_ids": [list(pos) for _, pos, _ in QUERIES],
        "negative_doc_ids": [list(neg) for _, _, neg in QUERIES],
        "negative_mask": [[1, 1] for _ in QUERIES],
    }
    allpos_q, allpos_pos, allpos_neg, allpos_w, allpos_g = [], [], [], [], []
    for gid, (query, pos, neg) in enumerate(QUERIES):
        w = 1.0 / len(pos)
        for p in pos:
            allpos_q.append(query)
            allpos_pos.append(p)
            allpos_neg.append(list(neg))
            allpos_w.append(w)
            allpos_g.append(gid)
    allpos = {
        "query": allpos_q,
        "positive_doc_id": allpos_pos,
        "negative_doc_ids": allpos_neg,
        "negative_mask": [[1, 1] for _ in allpos_q],
        "sample_weight": allpos_w,
        "query_group_id": allpos_g,
    }

    hardneg = ROOT / "hardneg"
    Dataset.from_dict(judged).save_to_disk(str(hardneg / "judged"))
    Dataset.from_dict(allpos).save_to_disk(str(hardneg / "allpos"))
    print(f"wrote {ROOT / 'part_000.parquet'} ({len(PAGES)} pages)")
    print(f"wrote {hardneg / 'judged'} ({len(QUERIES)} queries)")
    print(f"wrote {hardneg / 'allpos'} ({len(allpos_q)} rows)")


if __name__ == "__main__":
    main()
