#!/usr/bin/env python3
"""Independent MaxSim on a dump (raw) or HAC index.

Does not re-encode. Reports NDCG / Recall / MAP / MRR at @1/@5/@10.
Search timing is query-already-encoded MaxSim only — not product QPS.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))

import torch

from common import (
    BUDGET,
    CEILING,
    DUMP_PROTOCOL,
    HEAD_DIM,
    MVT,
    REQUIRED_KS,
    REQUIRED_METRICS,
    SOUP_NAME,
    WEIGHT,
    atomic_write_json,
    dump_ok,
    gib_per_million_pages,
    has_complete_metrics,
    load_qrels,
    padded_from_pack,
    w_tag,
)

import eval as evie_eval  # noqa: E402
from paths import forbid_venv_path  # noqa: E402


DS_LABEL = evie_eval.DS_LABEL
METRIC_NAMES = evie_eval.METRIC_NAMES


def load_task_list(dump_root: Path) -> list[dict]:
    tasks_dir = dump_root / "tasks"
    recs = []
    for path in sorted(tasks_dir.glob("*.json")):
        if dump_ok(path, dump_root):
            recs.append(json.loads(path.read_text(encoding="utf-8")))
    if not recs:
        raise SystemExit(f"no complete dump tasks under {tasks_dir}")
    return recs


def load_queries(dump_root: Path, rec: dict) -> torch.Tensor:
    blob = torch.load(dump_root / rec["queries"], map_location="cpu", weights_only=False)
    return padded_from_pack(blob["packed"], blob["offsets"])


def load_raw_docs(dump_root: Path, rec: dict) -> torch.Tensor:
    blob = torch.load(dump_root / rec["docs"], map_location="cpu", weights_only=False)
    return padded_from_pack(blob["packed"], blob["offsets"])


def load_clustered_docs(cluster_root: Path, rec: dict, tag: str, budget: int) -> tuple[torch.Tensor, dict]:
    path = Path(cluster_root) / tag / "docs" / f"{rec['corpus_id']}.pt"
    blob = torch.load(path, map_location="cpu", weights_only=False)
    padded = blob["padded"]
    if padded.ndim != 3 or int(padded.shape[-1]) != HEAD_DIM:
        raise RuntimeError(f"bad clustered tensor {path}: {tuple(padded.shape)}")
    if int(padded.shape[1]) != int(budget):
        raise RuntimeError(f"clustered K {padded.shape[1]} != budget {budget} at {path}")
    return padded, blob.get("stats") or {}


def score_task(q_emb, d_emb, relevant, graded, device, chunk_q, chunk_d):
    q = q_emb.to(device)
    t0 = time.perf_counter()
    scores = evie_eval.chunked_maxsim(q, d_emb, chunk_q=chunk_q, chunk_d=chunk_d)
    dt = time.perf_counter() - t0
    metrics = evie_eval.metrics_from_scores(scores, relevant, graded, list(REQUIRED_KS))
    n_q = int(q_emb.shape[0])
    return metrics, {
        "n_docs": int(d_emb.shape[0]),
        "n_queries": n_q,
        "doc_tokens": int(d_emb.shape[1]),
        "query_tokens_padded": int(q_emb.shape[1]),
        "search_sec": round(dt, 4),
        "search_ms_per_query": round(1000.0 * dt / max(n_q, 1), 4),
        "encode_ms_per_query": None,
        "note": "pure search; query vectors already dumped; not product QPS",
        "chunk_q": chunk_q,
        "chunk_d": chunk_d,
        "device": str(device),
    }


def write_summary(out_dir: Path, recs: list[dict], expected: list[dict], args, mode: str, w):
    by_ds = defaultdict(list)
    failed = [r for r in recs if "error" in r]
    ok = [r for r in recs if "metrics" in r and has_complete_metrics(r["metrics"])]
    for rec in ok:
        by_ds[rec["dataset"]].append(rec)
    expected_by_ds = defaultdict(int)
    for rec in expected:
        expected_by_ds[rec["dataset"]] += 1
    expected_tags = {rec["task"] for rec in expected}
    missing = sorted(expected_tags - {r["task"] for r in ok})

    def avg_metric(rows, name):
        vals = [r["metrics"].get(name) for r in rows]
        vals = [v for v in vals if v is not None]
        return (sum(vals) / len(vals)) if vals else None

    ks = list(REQUIRED_KS)
    per_ds = {}
    for ds, rows in by_ds.items():
        entry = {f"{m}@{k}": avg_metric(rows, f"{m}@{k}") for m in METRIC_NAMES for k in ks}
        entry["n_subsets"] = len(rows)
        entry["n_subsets_expected"] = expected_by_ds.get(ds, len(rows))
        entry["n_queries"] = sum(r["info"]["n_queries"] for r in rows)
        entry["n_docs"] = sum(r["info"]["n_docs"] for r in rows)
        search = [r["info"]["search_ms_per_query"] for r in rows if r["info"].get("search_ms_per_query") is not None]
        entry["search_ms_per_query_mean_unweighted"] = (
            sum(search) / len(search) if search else None
        )
        per_ds[ds] = entry

    headline = {
        "V1 nDCG@10": per_ds.get("eval", {}).get("ndcg@10"),
        "V2 nDCG@10": per_ds.get("eval_v2", {}).get("ndcg@10"),
        "V3 nDCG@10": per_ds.get("eval_v3", {}).get("ndcg@10"),
        "JinaVDR nDCG@10": per_ds.get("JinaVDR", {}).get("ndcg@10"),
    }
    four = [headline[k] for k in ("V1 nDCG@10", "V2 nDCG@10", "V3 nDCG@10", "JinaVDR nDCG@10")]
    averages = {
        "avg4_ndcg@10": (sum(four) / 4 if all(v is not None for v in four) else None),
    }
    complete = not failed and not missing
    vec_per_page = 1.0  # overwritten below
    if mode == "k":
        vec_per_page = float(args.budget)
        storage_note = f"independent K{int(args.budget)} candidate layer, no context, no raw rerank"
    else:
        vec_per_page = None
        storage_note = "raw full sequence from dump; not a compressed index"
    summary = {
        "status": "complete" if complete else "partial",
        "run_name": args.run_name,
        "mode": mode,
        "position_weight": None if mode == "raw" else float(w),
        "budget": None if mode == "raw" else int(args.budget),
        "export_context": False,
        "head_dim": HEAD_DIM,
        "max_visual_tokens": MVT,
        "bidirectional_attention": "on",
        "weights": SOUP_NAME,
        "alpha": 0.5,
        "dump_protocol": DUMP_PROTOCOL,
        "eval_protocol": evie_eval.EVAL_PROTOCOL,
        "ks": ks,
        "expected_tasks": len(expected),
        "completed_tasks": len(ok),
        "n_failed": len(failed),
        "failed_tasks": failed,
        "missing_tasks": missing,
        "headline": headline,
        "averages": averages,
        "per_dataset": per_ds,
        "per_task": ok,
        "ceiling": CEILING,
        "storage": {
            "vectors_per_page": vec_per_page,
            "dim": HEAD_DIM,
            "bytes_per_vec": 2,
            "gib_per_million_pages": (
                None if vec_per_page is None else round(gib_per_million_pages(vec_per_page), 4)
            ),
            "context": False,
            "two_stage": False,
            "note": storage_note,
        },
        "search": {
            "kind": "independent_pure_search",
            "includes_query_encode": False,
            "not_product_qps": True,
            "two_stage": False,
        },
    }
    atomic_write_json(out_dir / "summary.json", summary)
    lines = [
        f"# {args.run_name} {mode}"
        + (f" K{int(args.budget)} {w_tag(w)}" if mode == "k" else ""),
        "",
        f"status `{summary['status']}` · {summary['completed_tasks']}/{summary['expected_tasks']}",
        "",
        "Independent retrieval. Queries are already dumped; ms below is MaxSim only, not product QPS.",
        "",
        "| board | nDCG@1 | nDCG@5 | nDCG@10 | Recall@1 | Recall@5 | Recall@10 | MAP@1 | MAP@5 | MAP@10 | MRR@1 | MRR@5 | MRR@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def pct(v):
        return "—" if v is None else f"{v * 100:.2f}"

    for ds, label in DS_LABEL.items():
        e = per_ds.get(ds)
        if not e:
            continue
        cells = " | ".join(
            pct(e.get(name))
            for name in (
                "ndcg@1",
                "ndcg@5",
                "ndcg@10",
                "recall@1",
                "recall@5",
                "recall@10",
                "map@1",
                "map@5",
                "map@10",
                "mrr@1",
                "mrr@5",
                "mrr@10",
            )
        )
        lines.append(f"| {label} | {cells} |")
    lines += [
        "",
        f"uncompressed d{HEAD_DIM} eval V3 nDCG@10 = {CEILING['V3 nDCG@10']} "
        f"(source `{CEILING['source']}`)",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"[eval] {summary['status']} {summary['completed_tasks']}/{summary['expected_tasks']} "
        f"V3 nDCG@10={headline.get('V3 nDCG@10')}",
        flush=True,
    )
    if not complete:
        raise SystemExit(2)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--cluster-root", default="")
    ap.add_argument("--mode", choices=("raw", "k"), required=True)
    ap.add_argument("--w", type=float, default=WEIGHT)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--run-name", default="evie-compress")
    ap.add_argument("--chunk-q", type=int, default=64)
    ap.add_argument("--chunk-d", type=int, default=256)
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.mode == "k":
        if args.w is None:
            raise SystemExit("k mode requires --w")
        if not args.cluster_root:
            raise SystemExit("k mode requires --cluster-root")
        tag = w_tag(args.w)
    else:
        tag = "raw"
        args.w = None

    rank, world, local = evie_eval.setup_ddp()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    dump_root = forbid_venv_path(args.dump_dir, "dump-dir")
    out_dir = forbid_venv_path(args.output_dir, "output-dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = load_task_list(dump_root)

    cached = {}
    if args.resume:
        cached = evie_eval.load_cached_task_map(out_dir, protocol=evie_eval.EVAL_PROTOCOL)
        cached = {k: v for k, v in cached.items() if has_complete_metrics(v.get("metrics"))}

    pending = [rec for rec in expected if rec["task"] not in cached]
    if world > 1:
        # greedy by query*doc; corpora are already on disk so V3 langs need not colocate
        pending = sorted(pending, key=lambda r: -(int(r["n_docs"]) * int(r["n_queries"])))
        pending = pending[rank::world]
    print(f"[eval rank{rank}] {args.mode} {tag} pending {len(pending)}/{len(expected)} on {device}", flush=True)

    results = []
    for rec in pending:
        tag_task = rec["task"]
        t0 = time.time()
        try:
            q_emb = load_queries(dump_root, rec)
            if args.mode == "raw":
                d_emb = load_raw_docs(dump_root, rec)
            else:
                d_emb, _stats = load_clustered_docs(Path(args.cluster_root), rec, tag, args.budget)
            relevant, graded = load_qrels(dump_root / rec["qrels"])
            if len(relevant) != q_emb.shape[0]:
                raise RuntimeError(
                    f"{tag_task}: qrels {len(relevant)} != queries {q_emb.shape[0]}"
                )
            if d_emb.shape[0] != rec["n_docs"]:
                raise RuntimeError(
                    f"{tag_task}: docs {d_emb.shape[0]} != dump n_docs {rec['n_docs']}"
                )
            metrics, info = score_task(
                q_emb, d_emb, relevant, graded, device, args.chunk_q, args.chunk_d
            )
            row = {
                "task": tag_task,
                "dataset": rec["dataset"],
                "subset": rec["subset"],
                "lang": rec.get("lang"),
                "fmt": rec["fmt"],
                "metrics": metrics,
                "info": info,
                "seconds": round(time.time() - t0, 1),
                "eval_protocol": evie_eval.EVAL_PROTOCOL,
                "dump_protocol": DUMP_PROTOCOL,
                "run_name": args.run_name,
                "mode": args.mode,
                "position_weight": args.w,
            }
            evie_eval.save_task_result(out_dir, row, run_name=args.run_name, contract_id=DUMP_PROTOCOL)
            results.append(row)
            print(
                f"[eval rank{rank}] {tag_task}: ndcg@10={metrics['ndcg@10']*100:.2f} "
                f"{info['n_queries']}q/{info['n_docs']}d search={info['search_ms_per_query']:.2f}ms/q",
                flush=True,
            )
        except Exception as exc:
            import traceback

            traceback.print_exc()
            results.append({"task": tag_task, "dataset": rec["dataset"], "error": repr(exc)})
            print(f"[eval rank{rank}] {tag_task} FAILED: {exc!r}", flush=True)

    atomic_write_json(out_dir / f"rank_{rank}.json", results)
    if world > 1:
        torch.distributed.destroy_process_group()
    if rank != 0:
        return

    deadline = time.monotonic() + int(os.environ.get("EVAL_FINALIZE_TIMEOUT_S", "21600"))
    missing_ranks = list(range(world))
    last = 0.0
    while missing_ranks:
        missing_ranks = []
        for r in range(world):
            p = out_dir / f"rank_{r}.json"
            try:
                json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                missing_ranks.append(r)
        if not missing_ranks:
            break
        now = time.monotonic()
        if now >= deadline:
            print(f"[eval] finalize timeout; missing {missing_ranks}", flush=True)
            break
        if now - last >= 60:
            print(f"[eval] waiting for ranks {missing_ranks}", flush=True)
            last = now
        time.sleep(2)

    merged = evie_eval.load_cached_task_map(out_dir, protocol=evie_eval.EVAL_PROTOCOL)
    rows = []
    failed = []
    for r in range(world):
        p = out_dir / f"rank_{r}.json"
        if not p.exists():
            continue
        for rec in json.loads(p.read_text(encoding="utf-8")):
            if "error" in rec:
                failed.append(rec)
            elif rec.get("task") in merged:
                rows.append(merged[rec["task"]])
    # durable cache wins
    rows = [merged[rec["task"]] for rec in expected if rec["task"] in merged]
    for miss in failed:
        if miss["task"] not in merged:
            rows.append(miss)
    write_summary(out_dir, rows, expected, args, args.mode, args.w)


if __name__ == "__main__":
    main()
