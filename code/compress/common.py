#!/usr/bin/env python3
"""Shipped HAC recipe: d64 × K32. Change HEAD_DIM / BUDGET here to match other published SKUs."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch

HEAD_DIM = 64
BUDGET = 32
MVT = 1024
MERGE_SIZE = 2
WEIGHT = 0.1
DUMP_PROTOCOL = "evie-hac-d64-k32-mvt1024-imagespan"
REQUIRED_KS = (1, 5, 10)
REQUIRED_METRICS = tuple(
    f"{metric}@{k}"
    for metric in ("recall", "ndcg", "mrr", "map")
    for k in REQUIRED_KS
)

_raw_model = os.environ.get("EVIE_4_5B_DIR") or os.environ.get("MODEL_DIR") or ""
MODEL_DIR = Path(_raw_model) if _raw_model else Path()
SOUP_DIR = MODEL_DIR
SOUP_NAME = MODEL_DIR.name if _raw_model else "EVIE-4.5B"

CEILING = {
    "source": "EVIE-4.5B uncompressed d64",
    "V1 nDCG@10": 91.77,
    "V2 nDCG@10": 70.83,
    "V3 nDCG@10": 63.56,
    "JinaVDR nDCG@10": 80.77,
    "Avg4 nDCG@10": 76.73,
}


def w_tag(w: float = WEIGHT) -> str:
    return f"w{int(round(float(w) * 100)):02d}"


def gib_per_million_pages(vec_per_page: float = BUDGET, dim: int = HEAD_DIM, nbytes: int = 2) -> float:
    return float(vec_per_page) * dim * nbytes * 1_000_000 / (2**30)


def atomic_write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def atomic_torch_save(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def pack_seqs(seqs: list[torch.Tensor]) -> dict:
    if not seqs:
        raise ValueError("empty sequence list")
    dim = int(seqs[0].shape[-1])
    lengths = [int(s.shape[0]) for s in seqs]
    packed = torch.empty(sum(lengths), dim, dtype=torch.float16)
    offsets = torch.zeros(len(seqs) + 1, dtype=torch.int64)
    pos = 0
    for i, seq in enumerate(seqs):
        if seq.ndim != 2 or int(seq.shape[-1]) != dim:
            raise ValueError(f"seq {i} shape {tuple(seq.shape)} != [L,{dim}]")
        n = int(seq.shape[0])
        packed[pos : pos + n] = seq.detach().to(dtype=torch.float16, device="cpu")
        pos += n
        offsets[i + 1] = pos
    return {"packed": packed, "offsets": offsets, "dim": dim, "n": len(seqs)}


def padded_from_pack(packed: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    n = int(offsets.numel()) - 1
    if n <= 0:
        return packed.new_zeros((0, 0, packed.shape[-1]))
    lengths = (offsets[1:] - offsets[:-1]).tolist()
    lmax = max(lengths)
    out = packed.new_zeros((n, lmax, packed.shape[-1]))
    for i, length in enumerate(lengths):
        a = int(offsets[i])
        if length:
            out[i, :length] = packed[a : a + length]
    return out


def dump_ok(task_json: Path, dump_root: Path) -> bool:
    try:
        rec = json.loads(Path(task_json).read_text(encoding="utf-8"))
    except Exception:
        return False
    if rec.get("dump_protocol") != DUMP_PROTOCOL:
        return False
    docs = dump_root / rec["docs"]
    queries = dump_root / rec["queries"]
    qrels = dump_root / rec["qrels"]
    return docs.is_file() and queries.is_file() and qrels.is_file()


def load_qrels(path: Path) -> tuple[list[set[int]], list[dict[int, float]]]:
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    relevant = [set(int(x) for x in row) for row in rec["relevant"]]
    graded = [{int(k): float(v) for k, v in row.items()} for row in rec["graded"]]
    return relevant, graded


def dump_qrels(path: Path, relevant, graded) -> None:
    payload = {
        "relevant": [sorted(int(x) for x in row) for row in relevant],
        "graded": [{str(int(k)): float(v) for k, v in row.items()} for row in graded],
    }
    atomic_write_json(path, payload)


def has_complete_metrics(metrics) -> bool:
    if not isinstance(metrics, dict):
        return False
    return all(
        isinstance(metrics.get(name), (int, float)) and math.isfinite(float(metrics[name]))
        for name in REQUIRED_METRICS
    )
