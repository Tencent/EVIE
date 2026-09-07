#!/usr/bin/env python3
"""Encode EVIE-4.5B (active head = HEAD_DIM) once: image-span tokens + queries.

Query is not pooled. Resume skips task json files whose dump_protocol still matches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))

import torch
from torch.utils.data import DataLoader

from common import (
    DUMP_PROTOCOL,
    HEAD_DIM,
    MVT,
    SOUP_DIR,
    SOUP_NAME,
    atomic_torch_save,
    atomic_write_json,
    dump_ok,
    dump_qrels,
    pack_seqs,
)

import eval as evie_eval  # noqa: E402
from hac import image_span_from_mask  # noqa: E402
from paths import forbid_venv_path  # noqa: E402
from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor  # noqa: E402
from colpali_engine.models.qwen3_5.colqwen3_5.modeling_colqwen3_5 import (  # noqa: E402
    set_active_head,
)
from transformers.models.qwen3_5 import Qwen3_5Config  # noqa: E402


def corpus_id_for(task: dict, max_docs: int) -> str:
    if task["fmt"] == "beir":
        payload = "beir|" + "\0".join(task["corpus"])
    else:
        payload = "paired|" + "\0".join(task["paired"])
    payload += f"|max_docs={int(max_docs)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _first(meta: dict, *keys, default):
    for key in keys:
        value = meta.get(key)
        if value is not None:
            return value
    return default


def split_doc_batch(emb: torch.Tensor, batch: dict, processor) -> list[dict]:
    image_token_id = int(processor.image_token_id)
    merge = int(processor.image_processor.merge_size)
    attn = batch["attention_mask"].bool()
    ids = batch["input_ids"]
    thw = batch["image_grid_thw"]
    recs = []
    for i in range(emb.shape[0]):
        mask = attn[i]
        vec = emb[i][mask].contiguous().half()
        tok = ids[i][mask]
        start, end = image_span_from_mask(tok == image_token_id)
        grid = [int(x) for x in thw[i].tolist()]
        expected = (grid[1] // merge) * (grid[2] // merge)
        if (end - start) != expected:
            raise RuntimeError(
                f"image tokens {end - start} != grid {grid} merge={merge} expected={expected}"
            )
        recs.append(
            {
                "vec": vec.cpu(),
                "image_start": int(start),
                "image_end": int(end),
                "grid_thw": grid,
            }
        )
    return recs


@torch.no_grad()
def encode_docs(processor, model, paths, device, batch_size, num_workers, meta_keys, max_docs, dedupe_key=None):
    ds = evie_eval._ImageParquetDataset(
        paths, meta_keys=meta_keys, skip_empty_query=False, dedupe_key=dedupe_key
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=evie_eval._collate_pil,
        prefetch_factor=6 if num_workers > 0 else None,
        persistent_workers=False,
    )
    records = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = None
        it = iter(loader)

        def submit_next():
            try:
                imgs, metas = next(it)
            except StopIteration:
                return None
            return pool.submit(processor.process_images, imgs), metas

        pending = submit_next()
        while pending is not None:
            fut, metas = pending
            batch = fut.result()
            nxt = submit_next()
            batch_cpu = batch
            batch_gpu = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            emb = model(**batch_gpu).float().cpu()
            if emb.ndim != 3 or int(emb.shape[-1]) != HEAD_DIM:
                raise RuntimeError(f"doc emb shape {tuple(emb.shape)} is not [B,L,{HEAD_DIM}]")
            recs = split_doc_batch(emb, batch_cpu, processor)
            for rec, meta in zip(recs, metas):
                rec["meta"] = meta
                records.append(rec)
            if max_docs and len(records) >= max_docs:
                records = records[:max_docs]
                break
            pending = nxt
    return records


@torch.no_grad()
def encode_queries(processor, model, texts, device, batch_size):
    seqs = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        chunk = [t if (t and str(t).strip()) else " " for t in chunk]
        batch = processor.process_queries(chunk)
        attn = batch["attention_mask"].bool()
        batch_gpu = {k: v.to(device) for k, v in batch.items()}
        for obj in (model, getattr(model, "get_base_model", lambda: None)()):
            if obj is not None and hasattr(obj, "rope_deltas"):
                obj.rope_deltas = None
        emb = model(**batch_gpu).float().cpu()
        if emb.ndim != 3 or int(emb.shape[-1]) != HEAD_DIM:
            raise RuntimeError(f"query emb shape {tuple(emb.shape)} is not [B,L,{HEAD_DIM}]")
        for i in range(emb.shape[0]):
            seqs.append(emb[i][attn[i]].contiguous().half())
    return seqs


def save_docs(path: Path, records: list[dict], ids: list[str]) -> dict:
    packed = pack_seqs([r["vec"] for r in records])
    payload = {
        "ids": ids,
        "packed": packed["packed"],
        "offsets": packed["offsets"],
        "image_start": torch.tensor([r["image_start"] for r in records], dtype=torch.int64),
        "image_end": torch.tensor([r["image_end"] for r in records], dtype=torch.int64),
        "grid_thw": torch.tensor([r["grid_thw"] for r in records], dtype=torch.int32),
        "dim": HEAD_DIM,
        "n": len(records),
        "dump_protocol": DUMP_PROTOCOL,
    }
    atomic_torch_save(path, payload)
    n_image = [int(r["image_end"] - r["image_start"]) for r in records]
    n_full = [int(r["vec"].shape[0]) for r in records]
    return {
        "n_docs": len(records),
        "avg_full_tokens": sum(n_full) / len(n_full),
        "avg_image_tokens": sum(n_image) / len(n_image),
        "min_image_tokens": min(n_image),
        "max_image_tokens": max(n_image),
    }


def load_model(device):
    if not str(SOUP_DIR) or not (SOUP_DIR / "model.safetensors").is_file():
        raise FileNotFoundError("set MODEL_DIR or EVIE_4_5B_DIR to an EVIE-4.5B checkpoint")
    model_source = str(SOUP_DIR)
    processor = ColQwen3_5Processor.from_pretrained(model_source, max_num_visual_tokens=MVT)
    config = Qwen3_5Config.from_pretrained(model_source)
    run_path = SOUP_DIR / "run_config.json"
    if run_path.is_file():
        run_cfg = json.loads(run_path.read_text(encoding="utf-8"))
        heads = [int(d) for d in run_cfg["head_dims"]]
    else:
        heads = [int(d) for d in (getattr(config, "head_dims", None) or [])]
        if not heads:
            raise FileNotFoundError(
                f"{SOUP_DIR} has no head_dims; point MODEL_DIR at EVIE-4.5B"
            )
    config.head_dims = heads
    config.dim = max(heads)
    attn = os.environ.get("EVAL_ATTN", "flash_attention_2")
    model = ColQwen3_5.from_pretrained(
        model_source,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn,
    )
    model.enable_bidirectional_attention()
    set_active_head(model, HEAD_DIM)
    model = model.to(device).eval()
    merge = int(processor.image_processor.merge_size)
    if merge != 2:
        raise RuntimeError(f"expected merge_size=2, got {merge}")
    print(
        f"[dump] model={SOUP_NAME} head=d{HEAD_DIM} mvt={MVT} bidir=on "
        f"processor={model_source} device={device}",
        flush=True,
    )
    return processor, model


def dump_paired(task, processor, model, device, args, dump_root: Path, cid: str):
    docs_path = dump_root / "docs" / f"{cid}.pt"
    if not docs_path.is_file():
        records = encode_docs(
            processor,
            model,
            task["paired"],
            device,
            args.embed_batch,
            args.num_workers,
            meta_keys=["image_filename"],
            max_docs=args.max_docs,
            dedupe_key="image_filename",
        )
        if not records:
            raise ValueError(f"no candidate pages for {evie_eval.task_tag(task)}")
        ids = []
        page_to_idx = {}
        for i, rec in enumerate(records):
            key = rec["meta"].get("image_filename")
            if key is None:
                raise ValueError(f"paired candidate missing image_filename in {task['subset']}")
            key = str(key)
            if key in page_to_idx:
                raise ValueError(f"duplicate page was encoded twice: {key}")
            page_to_idx[key] = i
            ids.append(key)
        stats = save_docs(docs_path, records, ids)
    else:
        blob = torch.load(docs_path, map_location="cpu", weights_only=False)
        ids = list(blob["ids"])
        page_to_idx = {k: i for i, k in enumerate(ids)}
        stats = {"n_docs": len(ids), "reused": True}

    queries, golds = evie_eval._paired_query_rows(task["paired"], page_to_idx)
    if not queries:
        raise ValueError(f"no non-empty queries for {task['subset']}")
    if args.max_queries and args.max_queries < len(queries):
        queries = queries[: args.max_queries]
        golds = golds[: args.max_queries]
    q_seqs = encode_queries(processor, model, queries, device, args.embed_batch)
    relevant = [{g} for g in golds]
    graded = [{g: 1.0} for g in golds]
    return stats, q_seqs, relevant, graded


def dump_beir(task, processor, model, device, args, dump_root: Path, cid: str, corpus_cache: dict):
    docs_path = dump_root / "docs" / f"{cid}.pt"
    cache_key = (cid, args.max_docs)
    if corpus_cache.get("key") == cache_key:
        ids = corpus_cache["ids"]
        stats = {"n_docs": len(ids), "reused": True, "in_memory": True}
    elif docs_path.is_file():
        blob = torch.load(docs_path, map_location="cpu", weights_only=False)
        ids = list(blob["ids"])
        corpus_cache.clear()
        corpus_cache.update(key=cache_key, ids=ids)
        stats = {"n_docs": len(ids), "reused": True}
    else:
        records = encode_docs(
            processor,
            model,
            task["corpus"],
            device,
            args.embed_batch,
            args.num_workers,
            meta_keys=["corpus-id", "id"],
            max_docs=args.max_docs,
        )
        ids = [str(_first(r["meta"], "corpus-id", "id", default=f"c{i}")) for i, r in enumerate(records)]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate corpus ids in {task['subset']}")
        stats = save_docs(docs_path, records, ids)
        corpus_cache.clear()
        corpus_cache.update(key=cache_key, ids=ids)

    cid_to_idx = {cid_: i for i, cid_ in enumerate(ids)}
    q_rows = evie_eval.read_parquet_rows(task["queries"])
    q_texts = [str(r["text"] if "text" in r else r["query"]) for r in q_rows]
    q_ids = [str(r.get("id", r.get("query-id", f"q{i}"))) for i, r in enumerate(q_rows)]
    rel_by_q, grad_by_q = defaultdict(set), defaultdict(dict)
    for row in evie_eval.read_parquet_rows(task["qrels"]):
        qid = str(row.get("query-id", row.get("id", "")))
        doc = str(row.get("corpus-id", row.get("id", "")))
        score = float(row.get("score", 1.0) or 1.0)
        if score > 0 and doc in cid_to_idx:
            rel_by_q[qid].add(cid_to_idx[doc])
            grad_by_q[qid][cid_to_idx[doc]] = score
    keep = [
        i
        for i in range(len(q_ids))
        if q_texts[i].strip()
        and q_texts[i].lower() != "none"
        and q_ids[i] in rel_by_q
        and rel_by_q[q_ids[i]]
    ]
    if args.max_queries and args.max_queries < len(keep):
        keep = keep[: args.max_queries]
    if not keep:
        raise ValueError(f"no non-empty queries with valid qrels for {task['subset']}")
    q_seqs = encode_queries(processor, model, [q_texts[i] for i in keep], device, args.embed_batch)
    relevant = [rel_by_q[q_ids[i]] for i in keep]
    graded = [grad_by_q[q_ids[i]] for i in keep]
    return stats, q_seqs, relevant, graded


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--embed-batch", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--datasets", default="")
    return ap.parse_args()


def main():
    args = parse_args()
    rank, world, local = evie_eval.setup_ddp()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    dump_root = forbid_venv_path(args.dump_dir, "dump-dir")
    dump_root.mkdir(parents=True, exist_ok=True)

    tasks = evie_eval.discover_tasks(args.eval_root)
    if args.datasets:
        want = {d.strip() for d in args.datasets.split(",") if d.strip()}
        tasks = [t for t in tasks if t["dataset"] in want]
    expected = len(tasks)
    if not args.datasets:
        want_full = {"eval": 10, "eval_v2": 4, "eval_v3": 48, "JinaVDR": 76}
        got = Counter(t["dataset"] for t in tasks)
        bad = {k: (want_full[k], got.get(k, 0)) for k in want_full if got.get(k, 0) != want_full[k]}
        if bad:
            raise RuntimeError(f"eval discovery incomplete: {bad}")

    pending = []
    for task in tasks:
        slug = evie_eval.task_slug(evie_eval.task_tag(task))
        marker = dump_root / "tasks" / f"{slug}.json"
        if dump_ok(marker, dump_root):
            continue
        pending.append(task)
    if world > 1:
        pending = evie_eval._balanced_shard(pending, world, rank)
    print(
        f"[dump rank{rank}] pending {len(pending)}/{expected} "
        f"(~{sum(evie_eval._task_corpus_size(t) for t in pending)} corpus rows)",
        flush=True,
    )

    results = []
    processor = model = None
    corpus_cache = {}
    if pending:
        processor, model = load_model(device)
    for task in pending:
        tag = evie_eval.task_tag(task)
        slug = evie_eval.task_slug(tag)
        cid = corpus_id_for(task, args.max_docs)
        t0 = time.time()
        try:
            if task["fmt"] == "paired":
                stats, q_seqs, relevant, graded = dump_paired(
                    task, processor, model, device, args, dump_root, cid
                )
            else:
                stats, q_seqs, relevant, graded = dump_beir(
                    task, processor, model, device, args, dump_root, cid, corpus_cache
                )
            q_path = dump_root / "queries" / f"{slug}.pt"
            r_path = dump_root / "qrels" / f"{slug}.json"
            atomic_torch_save(q_path, {**pack_seqs(q_seqs), "dump_protocol": DUMP_PROTOCOL})
            dump_qrels(r_path, relevant, graded)
            rec = {
                "task": tag,
                "slug": slug,
                "dataset": task["dataset"],
                "subset": task["subset"],
                "lang": task.get("lang"),
                "fmt": task["fmt"],
                "corpus_id": cid,
                "docs": f"docs/{cid}.pt",
                "queries": f"queries/{slug}.pt",
                "qrels": f"qrels/{slug}.json",
                "n_docs": int(stats["n_docs"]),
                "n_queries": len(q_seqs),
                "stats": stats,
                "seconds": round(time.time() - t0, 1),
                "dump_protocol": DUMP_PROTOCOL,
                "head_dim": HEAD_DIM,
                "max_visual_tokens": MVT,
                "bidirectional_attention": "on",
                "weights": SOUP_NAME,
                "alpha": 0.5,
            }
            atomic_write_json(dump_root / "tasks" / f"{slug}.json", rec)
            results.append(rec)
            print(
                f"[dump rank{rank}] {tag}: {rec['n_queries']}q / {rec['n_docs']}d "
                f"{rec['seconds']}s",
                flush=True,
            )
        except Exception as exc:
            import traceback

            traceback.print_exc()
            results.append({"task": tag, "slug": slug, "error": repr(exc)})
            print(f"[dump rank{rank}] {tag} FAILED: {exc!r}", flush=True)

    rank_path = dump_root / f"rank_{rank}.json"
    atomic_write_json(rank_path, results)
    if torch.cuda.is_available() and model is not None:
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"[dump rank{rank}] peak VRAM {peak:.2f} GiB", flush=True)

    if world > 1:
        torch.distributed.destroy_process_group()
    if rank != 0:
        return

    deadline = time.monotonic() + int(os.environ.get("EVAL_FINALIZE_TIMEOUT_S", "21600"))
    missing = list(range(world))
    last = 0.0
    while missing:
        missing = []
        for r in range(world):
            p = dump_root / f"rank_{r}.json"
            try:
                json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                missing.append(r)
        if not missing:
            break
        now = time.monotonic()
        if now >= deadline:
            print(f"[dump] finalize timeout; missing ranks {missing}", flush=True)
            break
        if now - last >= 60:
            print(f"[dump] waiting for ranks {missing}", flush=True)
            last = now
        time.sleep(2)

    ok = []
    failed = []
    for task in tasks:
        slug = evie_eval.task_slug(evie_eval.task_tag(task))
        marker = dump_root / "tasks" / f"{slug}.json"
        if dump_ok(marker, dump_root):
            ok.append(json.loads(marker.read_text(encoding="utf-8")))
        else:
            failed.append(evie_eval.task_tag(task))
    unique_docs = sorted({rec["corpus_id"] for rec in ok})
    done = {
        "dump_protocol": DUMP_PROTOCOL,
        "status": "complete" if not failed else "partial",
        "expected_tasks": expected,
        "completed_tasks": len(ok),
        "failed_or_missing": failed,
        "unique_corpora": len(unique_docs),
        "n_docs_sum_over_tasks": sum(rec["n_docs"] for rec in ok),
        "n_queries_sum": sum(rec["n_queries"] for rec in ok),
        "head_dim": HEAD_DIM,
        "max_visual_tokens": MVT,
        "weights": SOUP_NAME,
        "alpha": 0.5,
        "truncated": bool(args.max_docs or args.max_queries or args.datasets),
        "datasets": args.datasets or None,
        "eval_root": str(Path(args.eval_root).resolve()),
    }
    atomic_write_json(dump_root / "DONE.json", done)
    print(f"[dump] {done['status']} {done['completed_tasks']}/{expected} -> {dump_root / 'DONE.json'}", flush=True)
    if failed:
        for tag in failed[:20]:
            print(f"  - MISS {tag}", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
