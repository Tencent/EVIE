#!/usr/bin/env python3
"""HAC on a token dump. Image tokens only; no context; no re-encode.

K and dim come from common.py (shipped: K=32, d=64). Cluster assignment uses
sinusoidal position fusion; stored vectors are semantic means.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))

import torch

from common import (
    BUDGET,
    DUMP_PROTOCOL,
    HEAD_DIM,
    MERGE_SIZE,
    WEIGHT,
    atomic_torch_save,
    atomic_write_json,
    gib_per_million_pages,
    w_tag,
)
from paths import forbid_venv_path

_G_IMAGES = None
_G_GRIDS = None
_G_BUDGET = BUDGET
_G_W = WEIGHT
_G_MERGE = MERGE_SIZE


def _cluster_one(index: int):
    from hac import hac_pool

    pooled = hac_pool(
        _G_IMAGES[index],
        _G_GRIDS[index],
        budget=_G_BUDGET,
        position_weight=_G_W,
        merge_size=_G_MERGE,
    )
    return pooled.half().contiguous()


def load_image_pages(docs_path: Path):
    blob = torch.load(docs_path, map_location="cpu", weights_only=False)
    if blob.get("dump_protocol") not in (DUMP_PROTOCOL, None) and "packed" not in blob:
        raise RuntimeError(f"unrecognized dump: {docs_path}")
    packed = blob["packed"]
    offsets = blob["offsets"]
    starts = blob["image_start"]
    ends = blob["image_end"]
    grids = blob["grid_thw"]
    n = int(blob["n"])
    images = []
    grid_list = []
    for i in range(n):
        a, b = int(offsets[i]), int(offsets[i + 1])
        full = packed[a:b]
        images.append(full[int(starts[i]) : int(ends[i])].float())
        grid_list.append([int(x) for x in grids[i].tolist()])
    return blob.get("ids"), images, grid_list


def pad_pooled(pooled: list[torch.Tensor], budget: int, dim: int):
    n = len(pooled)
    out = torch.zeros(n, budget, dim, dtype=torch.float16)
    nvec = torch.zeros(n, dtype=torch.int64)
    for i, vec in enumerate(pooled):
        if vec.ndim != 2 or int(vec.shape[-1]) != dim:
            raise RuntimeError(f"pooled[{i}] shape {tuple(vec.shape)}")
        k = int(vec.shape[0])
        if k > budget:
            raise RuntimeError(f"pooled[{i}] has {k} > budget {budget}")
        out[i, :k] = vec.half()
        nvec[i] = k
    return out, nvec


def cluster_one_corpus(docs_path: Path, dest: Path, w: float, budget: int, workers: int, overwrite: bool):
    if dest.is_file() and not overwrite:
        print(f"[cluster] skip existing {dest}", flush=True)
        return json_sidecar_if_any(dest)
    ids, images, grids = load_image_pages(docs_path)
    n = len(images)
    global _G_IMAGES, _G_GRIDS, _G_BUDGET, _G_W, _G_MERGE
    _G_IMAGES = images
    _G_GRIDS = grids
    _G_BUDGET = int(budget)
    _G_W = float(w)
    _G_MERGE = MERGE_SIZE
    t0 = time.time()
    # Threads, not processes: sklearn releases the GIL, and fork-after-torch
    # deadlocks on this node (CUDA already initialized in the parent).
    if workers <= 1 or n <= 1:
        pooled = [_cluster_one(i) for i in range(n)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pooled = list(pool.map(_cluster_one, range(n), chunksize=8))
    padded, nvec = pad_pooled(pooled, budget, HEAD_DIM)
    dt = time.time() - t0
    stored = [int(x) for x in nvec.tolist()]
    stats = {
        "pages": n,
        "budget": int(budget),
        "position_weight": float(w),
        "merge_size": MERGE_SIZE,
        "export_context": False,
        "avg_image_tokens": sum(int(t.shape[0]) for t in images) / n,
        "avg_stored_tokens": sum(stored) / n,
        "min_stored_tokens": min(stored),
        "max_stored_tokens": max(stored),
        "pages_below_budget": sum(s < budget for s in stored),
        "gib_per_million_pages": round(gib_per_million_pages(sum(stored) / n), 4),
        "compress_sec": round(dt, 2),
        "workers": workers,
        "corpus_id": docs_path.stem,
    }
    payload = {
        "ids": ids,
        "padded": padded,
        "nvec": nvec,
        "dim": HEAD_DIM,
        "budget": int(budget),
        "position_weight": float(w),
        "export_context": False,
        "dump_protocol": DUMP_PROTOCOL,
        "stats": stats,
    }
    atomic_torch_save(dest, payload)
    atomic_write_json(dest.with_suffix(".json"), stats)
    print(
        f"[cluster] K={budget} w={w} {docs_path.name}: {n} pages "
        f"avg_stored={stats['avg_stored_tokens']:.2f} {dt:.1f}s -> {dest}",
        flush=True,
    )
    return stats


def json_sidecar_if_any(dest: Path):
    side = dest.with_suffix(".json")
    if side.is_file():
        return json.loads(side.read_text(encoding="utf-8"))
    return None


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--cluster-root", required=True)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    dump_root = forbid_venv_path(args.dump_dir, "dump-dir")
    forbid_venv_path(args.cluster_root, "cluster-root")
    docs_dir = dump_root / "docs"
    corpora = sorted(docs_dir.glob("*.pt"))
    if not corpora:
        raise SystemExit(f"no dumped corpora under {docs_dir}")
    weights = (WEIGHT,)
    workers = max(1, int(args.workers))
    print(
        f"[cluster] corpora={len(corpora)} w={WEIGHT} budget={args.budget} "
        f"workers={workers} no_context=1",
        flush=True,
    )
    summary = {"weights": {}, "dump_dir": str(dump_root), "budget": args.budget, "export_context": False}
    n_corpora = len(corpora)
    for w in weights:
        tag = w_tag(w)
        dest_dir = Path(args.cluster_root) / tag / "docs"
        dest_dir.mkdir(parents=True, exist_ok=True)
        done_path = Path(args.cluster_root) / tag / "DONE.json"
        existing_pts = list(dest_dir.glob("*.pt"))
        if (
            not args.overwrite
            and done_path.is_file()
            and len(existing_pts) == n_corpora
        ):
            rec = json.loads(done_path.read_text(encoding="utf-8"))
            if int(rec.get("n_corpora") or 0) == n_corpora:
                print(f"[cluster] skip {tag} already DONE n_corpora={n_corpora}", flush=True)
                summary["weights"][tag] = rec
                continue
        per = []
        for docs_path in corpora:
            dest = dest_dir / docs_path.name
            per.append(
                cluster_one_corpus(
                    docs_path, dest, w=w, budget=args.budget, workers=workers, overwrite=args.overwrite
                )
            )
        summary["weights"][tag] = {"position_weight": w, "n_corpora": len(per), "corpora": per}
        atomic_write_json(done_path, summary["weights"][tag])
    atomic_write_json(Path(args.cluster_root) / "DONE.json", summary)
    print(f"[cluster] done -> {Path(args.cluster_root) / 'DONE.json'}", flush=True)


if __name__ == "__main__":
    main()
