#!/usr/bin/env python3
"""Fold raw + HAC summaries into ladder.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    BUDGET,
    CEILING,
    DUMP_PROTOCOL,
    HEAD_DIM,
    REQUIRED_METRICS,
    SOUP_NAME,
    WEIGHT,
    atomic_write_json,
    gib_per_million_pages,
    has_complete_metrics,
)


def load_summary(path: Path):
    if not path.is_file():
        return None
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("status") != "complete":
        return rec
    for task in rec.get("per_task") or []:
        if not has_complete_metrics(task.get("metrics")):
            rec["status"] = "partial"
            rec.setdefault("invalid_metric_tasks", []).append(task.get("task"))
            return rec
    missing = [
        name
        for name in REQUIRED_METRICS
        if rec.get("per_dataset", {}).get("eval_v3") and rec["per_dataset"]["eval_v3"].get(name) is None
    ]
    if missing and rec.get("expected_tasks") == 138:
        rec["status"] = "partial"
    return rec


def headline_pct(summary, key):
    if not summary:
        return None
    v = (summary.get("headline") or {}).get(key)
    return None if v is None else round(v * 100, 2)


def pack(summary, path: Path):
    if summary is None:
        return None
    return {
        "status": summary.get("status"),
        "completed_tasks": summary.get("completed_tasks"),
        "expected_tasks": summary.get("expected_tasks"),
        "headline": summary.get("headline"),
        "averages": summary.get("averages"),
        "per_dataset": summary.get("per_dataset"),
        "storage": summary.get("storage"),
        "search": summary.get("search"),
        "path": str(path),
        "V3 nDCG@10": headline_pct(summary, "V3 nDCG@10"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", required=True)
    args = ap.parse_args()
    root = Path(args.eval_root)
    raw = load_summary(root / "raw" / "summary.json")
    hac = load_summary(root / f"k{BUDGET}" / "summary.json")
    v3 = headline_pct(hac, "V3 nDCG@10")
    ladder = {
        "dump_protocol": DUMP_PROTOCOL,
        "weights": SOUP_NAME,
        "head_dim": HEAD_DIM,
        "budget": BUDGET,
        "position_weight": WEIGHT,
        "export_context": False,
        "ceiling": CEILING,
        "raw_from_dump": pack(raw, root / "raw" / "summary.json"),
        "hac": pack(hac, root / f"k{BUDGET}" / "summary.json"),
        "sku": {
            "k": BUDGET,
            "d": HEAD_DIM,
            "w": WEIGHT,
            "bytes_per_page": BUDGET * HEAD_DIM * 2,
            "gib_per_million_pages": round(gib_per_million_pages(), 4),
            "V3 nDCG@10": v3,
        },
    }
    atomic_write_json(root / "ladder.json", ladder)
    print(json.dumps({"sku": ladder["sku"], "raw": ladder["raw_from_dump"] and ladder["raw_from_dump"]["V3 nDCG@10"]}, indent=2))


if __name__ == "__main__":
    main()
