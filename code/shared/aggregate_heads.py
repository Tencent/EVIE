#!/usr/bin/env python3
"""Aggregate per-head `d<k>/summary.json` into `summary_heads.json` / `.md`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FAMILIES = ("ndcg", "recall", "map", "mrr")
CUTOFFS = (1, 5, 10)
BOARDS = {
    "eval": "ViDoRe V1",
    "eval_v2": "ViDoRe V2",
    "eval_v3": "ViDoRe V3",
    "JinaVDR": "JinaVDR",
}
MISSING = "missing"


def board_means(summary: dict) -> dict[str, dict[str, float | str]]:
    buckets: dict[str, list[dict]] = {}
    for rec in summary.get("per_task", []):
        if rec.get("error"):
            continue
        buckets.setdefault(rec.get("dataset", "?"), []).append(rec.get("metrics") or {})

    out: dict[str, dict[str, float | str]] = {}
    for dataset, records in buckets.items():
        row: dict[str, float | str] = {"n_tasks": len(records)}
        for family in FAMILIES:
            for k in CUTOFFS:
                key = f"{family}@{k}"
                values = [m[key] for m in records if isinstance(m.get(key), (int, float))]
                if len(values) != len(records):
                    row[key] = MISSING
                    row.setdefault("_missing_reason", {})[key] = (
                        f"{len(records) - len(values)}/{len(records)} tasks lack {key}"
                    )
                else:
                    row[key] = round(100 * sum(values) / len(values), 2)
        out[dataset] = row
    return out


def fmt(value) -> str:
    return value if isinstance(value, str) else f"{value:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--heads", required=True)
    args = ap.parse_args()

    root = Path(args.eval_root)
    heads = [int(x) for x in args.heads.replace(" ", "").split(",") if x]
    payload: dict[str, dict] = {}
    for head in heads:
        path = root / f"d{head}" / "summary.json"
        if not path.is_file():
            payload[str(head)] = {"status": "missing", "reason": f"{path} not found"}
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        payload[str(head)] = {
            "status": summary.get("status"),
            "completed_tasks": summary.get("completed_tasks"),
            "expected_tasks": summary.get("expected_tasks"),
            "eval_contract_id": summary.get("eval_contract_id"),
            "headline": summary.get("headline"),
            "averages": summary.get("averages"),
            "boards": board_means(summary),
        }

    done = [h for h in heads if payload[str(h)].get("status") == "complete"]
    out_json = root / "summary_heads.json"
    out_json.write_text(
        json.dumps(
            {
                "heads": heads,
                "heads_complete": done,
                "note": "Per-head four-family x @1/@5/@10 means from each d<k>/summary.json.",
                "per_head": payload,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        f"# Matryoshka head eval ({len(done)}/{len(heads)} complete)",
        "",
        f"Root `{root}`. One `d<k>/summary.json` per head; this file only aggregates.",
        "",
    ]
    for dataset, board in BOARDS.items():
        present = [h for h in done if dataset in payload[str(h)]["boards"]]
        if not present:
            lines += [f"## {board}", "", f"missing: no head produced `{dataset}` tasks.", ""]
            continue
        n_tasks = payload[str(present[0])]["boards"][dataset]["n_tasks"]
        lines += [f"## {board} ({n_tasks} tasks)", ""]
        header = "| head | " + " | ".join(f"{f.upper()}@{k}" for f in FAMILIES for k in CUTOFFS) + " |"
        lines += [header, "|" + "---|" * (1 + len(FAMILIES) * len(CUTOFFS))]
        for head in present:
            row = payload[str(head)]["boards"][dataset]
            cells = [fmt(row[f"{f}@{k}"]) for f in FAMILIES for k in CUTOFFS]
            lines.append(f"| d{head} | " + " | ".join(cells) + " |")
        lines.append("")

    incomplete = [h for h in heads if h not in done]
    if incomplete:
        lines += ["## Incomplete", ""]
        for head in incomplete:
            rec = payload[str(head)]
            if rec.get("reason"):
                lines.append(f"- d{head}: {rec['reason']}")
            else:
                lines.append(
                    f"- d{head}: status={rec.get('status')}, "
                    f"{rec.get('completed_tasks')}/{rec.get('expected_tasks')} tasks"
                )
        lines.append("")

    (root / "summary_heads.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[aggregate] {len(done)}/{len(heads)} heads -> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
