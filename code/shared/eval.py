#!/usr/bin/env python3
"""ColQwen3.5 late-interaction eval: ViDoRe V1/V2/V3 + JinaVDR (paired + BEIR)."""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import math
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from peft import PeftModel
from torch.utils.data import DataLoader, IterableDataset
from transformers.models.qwen3_5 import Qwen3_5Config

from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor
from colpali_engine.models.qwen3_5.colqwen3_5.modeling_colqwen3_5 import set_active_head
from colpali_engine.utils.maxsim import maxsim_inbatch
from paths import forbid_venv_path


# Bump when paired/BEIR/query encoding rules change so stale task caches are ignored.
EVAL_PROTOCOL = "paired-all-pages-dedup+process_queries+ndcg2r-20260827"
REQUIRED_KS = (1, 5, 10)
REQUIRED_METRICS = tuple(
    f"{metric}@{k}"
    for metric in ("recall", "ndcg", "mrr", "map")
    for k in REQUIRED_KS
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def task_tag(task_or_rec):
    """Stable task id: ``dataset/subset`` or ``dataset/subset[lang]``."""
    if "task" in task_or_rec and isinstance(task_or_rec["task"], str):
        return task_or_rec["task"]
    tag = f"{task_or_rec['dataset']}/{task_or_rec['subset']}"
    lang = task_or_rec.get("lang")
    return f"{tag}[{lang}]" if lang else tag


def task_slug(tag: str) -> str:
    """Filesystem-safe name for per-task result files."""
    return tag.replace("/", "__").replace("[", "__").replace("]", "")


def tasks_dir(out_dir: Path) -> Path:
    return Path(out_dir) / "tasks"


def _has_complete_metrics(metrics) -> bool:
    if not isinstance(metrics, dict):
        return False
    return all(
        isinstance(metrics.get(name), (int, float))
        and math.isfinite(float(metrics[name]))
        for name in REQUIRED_METRICS
    )


def _is_ok_task_rec(
    rec,
    protocol: str | None = EVAL_PROTOCOL,
    contract_id: str | None = None,
) -> bool:
    if not isinstance(rec, dict) or "error" in rec or "metrics" not in rec:
        return False
    if not _has_complete_metrics(rec["metrics"]):
        return False
    if protocol is None:
        protocol_ok = True
    else:
        protocol_ok = rec.get("eval_protocol") == protocol
    contract_ok = contract_id is None or rec.get("eval_contract_id") == contract_id
    return protocol_ok and contract_ok


def save_task_result(
    out_dir: Path,
    rec: dict,
    run_name: str,
    contract_id: str,
) -> Path:
    """Persist one successful task under ``eval/tasks/`` (resume unit)."""
    d = tasks_dir(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    payload = dict(rec)
    payload["eval_protocol"] = EVAL_PROTOCOL
    payload["run_name"] = run_name
    payload["eval_contract_id"] = contract_id
    path = d / f"{task_slug(payload['task'])}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def load_cached_task_map(
    out_dir: Path,
    protocol: str = EVAL_PROTOCOL,
    contract_id: str | None = None,
) -> dict[str, dict]:
    """Load successful per-task results keyed by task tag."""
    d = tasks_dir(out_dir)
    out = {}
    if d.is_dir():
        for p in sorted(d.glob("*.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if _is_ok_task_rec(rec, protocol=protocol, contract_id=contract_id):
                out[rec["task"]] = rec
    return out


def to_pil(x):
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x))
    if isinstance(x, dict) and "bytes" in x:
        return Image.open(io.BytesIO(x["bytes"]))
    raise TypeError(f"cannot decode image of type {type(x)}")


def read_parquet_rows(paths):
    """Read small text-only parquet files (queries / qrels)."""
    rows = []
    for p in paths:
        rows.extend(pq.read_table(p).to_pylist())
    return rows


# --------------------------------------------------------------------------- #
# task discovery
# --------------------------------------------------------------------------- #
def _split_files(d):
    """Eval split only: some subsets ship train-*.parquet alongside test-*.parquet."""
    files = sorted(glob.glob(str(Path(d) / "*.parquet")))
    test = [f for f in files if Path(f).name.startswith("test-")]
    return test or files


def discover_tasks(eval_root: str):
    tasks = []
    root = Path(eval_root)
    for dataset in ("eval", "eval_v2", "eval_v3", "JinaVDR", "demo"):
        ddir = root / dataset
        if not ddir.is_dir():
            continue
        for sub in sorted(p for p in ddir.iterdir() if p.is_dir()):
            corpus_dirs = sorted(glob.glob(str(sub / "*-corpus")))
            if corpus_dirs:  # BEIR multi-language (vidore v3)
                # V3's six languages share the same page corpus. Canonicalize to
                # English (the frozen reproduce.py protocol) so it is encoded once.
                canonical_dir = sub / "english-corpus"
                if not canonical_dir.is_dir():
                    canonical_dir = Path(corpus_dirs[0])
                canonical_corpus = _split_files(canonical_dir)
                for cd in corpus_dirs:
                    lang = Path(cd).name.replace("-corpus", "")
                    qd = sub / f"{lang}-queries"
                    qrd = sub / f"{lang}-qrels"
                    if qd.is_dir() and qrd.is_dir():
                        tasks.append(
                            dict(
                                dataset=dataset, subset=sub.name, fmt="beir", lang=lang,
                                corpus=canonical_corpus,
                                queries=_split_files(qd),
                                qrels=_split_files(qrd),
                            )
                        )
                continue
            if (sub / "corpus").is_dir():  # BEIR single-language (vidore v2)
                tasks.append(
                    dict(
                        dataset=dataset, subset=sub.name, fmt="beir", lang=None,
                        corpus=_split_files(sub / "corpus"),
                        queries=_split_files(sub / "queries"),
                        qrels=_split_files(sub / "qrels") or _split_files(sub / "docs"),
                    )
                )
                continue
            paired = sorted(glob.glob(str(sub / "data" / "test-*.parquet")))
            if paired:  # PAIRED (vidore v1, JinaVDR)
                tasks.append(dict(dataset=dataset, subset=sub.name, fmt="paired", lang=None, paired=paired))
                continue
            for ld in sorted(p for p in sub.iterdir() if p.is_dir() and not p.name.startswith(".")):
                lang_files = sorted(glob.glob(str(ld / "test-*.parquet")))
                if lang_files:  # PAIRED per language (JinaVDR multilingual subsets)
                    tasks.append(
                        dict(dataset=dataset, subset=sub.name, fmt="paired",
                             lang=ld.name, paired=lang_files)
                    )
    return tasks


def _pad_cat(embs):
    """Right-pad variable-length (B, L, D) chunks and concat on batch dim."""
    if not embs:
        return torch.zeros((0, 0, 0))
    Lmax = max(e.shape[1] for e in embs)
    embs = [F.pad(e, (0, 0, 0, Lmax - e.shape[1])) for e in embs]
    return torch.cat(embs, 0)


class _ImageParquetDataset(IterableDataset):
    """Stream parquet image rows; decode to RGB PIL in DataLoader workers."""

    def __init__(
        self,
        paths,
        meta_keys=None,
        skip_empty_query=False,
        dedupe_key=None,
    ):
        self.paths = paths
        self.meta_keys = meta_keys or []
        self.skip_empty_query = skip_empty_query
        self.dedupe_key = dedupe_key

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (0, 1) if info is None else (info.id, info.num_workers)
        idx = 0
        seen = set()
        want_cols = ["image"] + list(self.meta_keys)
        if self.skip_empty_query and "query" not in want_cols:
            want_cols.append("query")
        if self.dedupe_key and self.dedupe_key not in want_cols:
            want_cols.append(self.dedupe_key)
        for p in self.paths:
            pf = pq.ParquetFile(p)
            avail = set(pf.schema_arrow.names)
            cols = [c for c in want_cols if c in avail]
            has_query = "query" in cols
            for rg in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg, columns=cols)
                img_col = tbl.column("image")
                meta_cols = {k: tbl.column(k) for k in self.meta_keys if k in avail}
                q_col = tbl.column("query") if has_query else None
                for j in range(tbl.num_rows):
                    if self.skip_empty_query and q_col is not None:
                        q = q_col[j].as_py()
                        q = "" if q is None else str(q).strip()
                        if not q or q.lower() == "none":
                            continue
                    dedupe_value = (
                        tbl.column(self.dedupe_key)[j].as_py()
                        if self.dedupe_key and self.dedupe_key in avail
                        else None
                    )
                    if dedupe_value is not None:
                        page_key = str(dedupe_value)
                        owner = int.from_bytes(
                            hashlib.blake2b(
                                page_key.encode("utf-8"), digest_size=8
                            ).digest(),
                            "little",
                        ) % nw
                        if owner != wid or page_key in seen:
                            continue
                        seen.add(page_key)
                    elif idx % nw != wid:
                        idx += 1
                        continue
                    idx += 1
                    img = to_pil(img_col[j].as_py()).convert("RGB")
                    meta = {k: meta_cols[k][j].as_py() if k in meta_cols else None for k in self.meta_keys}
                    yield (img, meta)


def _collate_pil(items):
    return [x[0] for x in items], [x[1] for x in items]


@torch.no_grad()
def embed_image_stream(
    processor, model, paths, device, batch_size, num_workers,
    meta_keys=None, max_docs=0, skip_empty_query=False, dedupe_key=None,
):
    """Embed images with CPU preprocess overlapped against GPU forward."""
    ds = _ImageParquetDataset(
        paths,
        meta_keys=meta_keys,
        skip_empty_query=skip_empty_query,
        dedupe_key=dedupe_key,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate_pil,
        prefetch_factor=6 if num_workers > 0 else None,
        persistent_workers=False,
    )
    out, metas, total = [], [], 0
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = None
        it = iter(loader)

        def submit_next():
            try:
                imgs, bm = next(it)
            except StopIteration:
                return None
            return (pool.submit(processor.process_images, imgs), bm, len(imgs))

        pending = submit_next()
        while pending is not None:
            fut, bm, n = pending
            batch = fut.result()
            nxt = submit_next()
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out.append(model(**batch).to("cpu"))
            metas.extend(bm)
            total += n
            if max_docs and total >= max_docs:
                break
            pending = nxt
    doc_emb = _pad_cat(out)
    if max_docs and doc_emb.shape[0] > max_docs:
        doc_emb = doc_emb[:max_docs]
        metas = metas[:max_docs]
    return doc_emb, metas


@torch.no_grad()
def embed_queries(processor, model, texts, device, batch_size):
    """Encode queries with ColQwen query prefix + 10 augmentation tokens.

    Must use ``process_queries`` (not ``process_texts``): training collator and
    standard ViDoRe eval append ``query_augmentation_token * 10``.
    """
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        chunk = [t if (t and t.strip()) else " " for t in chunk]
        batch = processor.process_queries(chunk)
        batch = {k: v.to(device) for k, v in batch.items()}
        # Clear rope cache before query forward (ColQwen hybrid-attn note).
        for obj in (model, getattr(model, "get_base_model", lambda: None)()):
            if obj is not None and hasattr(obj, "rope_deltas"):
                obj.rope_deltas = None
        out.append(model(**batch))
    return _pad_cat(out)


def chunked_maxsim(q_emb, doc_emb, chunk_q=64, chunk_d=256):
    """(Nq, Nd) MaxSim. Docs may live on CPU; each chunk is moved to the query device once."""
    dev = q_emb.device
    Nq, Nd = q_emb.shape[0], doc_emb.shape[0]
    scores = torch.zeros(Nq, Nd, device=dev, dtype=torch.float32)
    for di in range(0, Nd, chunk_d):
        d = doc_emb[di : di + chunk_d].to(dev, non_blocking=True).contiguous()
        for qi in range(0, Nq, chunk_q):
            q = q_emb[qi : qi + chunk_q].contiguous()
            scores[qi : qi + chunk_q, di : di + chunk_d] = maxsim_inbatch(q, d)
    return scores


METRIC_NAMES = ("recall", "ndcg", "mrr", "map")


def metrics_from_scores(scores, relevant, graded, ks):
    acc = {f"{m}@{k}": [] for k in ks for m in METRIC_NAMES}
    for q in range(scores.shape[0]):
        rel_set = relevant[q]
        if not rel_set:
            continue
        rel_grad = graded[q]
        order = scores[q].argsort(descending=True).tolist()
        for k in ks:
            topk = order[:k]
            acc[f"recall@{k}"].append(len(set(topk) & rel_set) / len(rel_set))

            dcg = sum(
                ((2 ** rel_grad.get(c, 0.0)) - 1) / math.log2(i + 2)
                for i, c in enumerate(topk)
                if rel_grad.get(c, 0.0) > 0
            )
            ideal = sorted(
                ((2 ** g) - 1 for g in rel_grad.values() if g > 0), reverse=True
            )[:k]
            idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
            acc[f"ndcg@{k}"].append(dcg / idcg if idcg > 0 else 0.0)

            acc[f"mrr@{k}"].append(
                next((1.0 / (i + 1) for i, c in enumerate(topk) if c in rel_set), 0.0)
            )

            hits, ap = 0, 0.0
            for i, c in enumerate(topk):
                if c in rel_set:
                    hits += 1
                    ap += hits / (i + 1)
            denom = min(len(rel_set), k)
            acc[f"map@{k}"].append(ap / denom if denom else 0.0)
    return {name: (sum(v) / len(v) if v else 0.0) for name, v in acc.items()}


# --------------------------------------------------------------------------- #
# runners
# --------------------------------------------------------------------------- #
def _paired_query_rows(paths, page_to_idx):
    """Read paired query metadata without loading the image column."""
    queries, golds = [], []
    for path in paths:
        pf = pq.ParquetFile(path)
        required = {"query", "image_filename"}
        missing = required - set(pf.schema_arrow.names)
        if missing:
            raise ValueError(f"paired parquet missing {sorted(missing)}: {path}")
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=["query", "image_filename"])
            for query, page in zip(
                table.column("query").to_pylist(),
                table.column("image_filename").to_pylist(),
            ):
                text = "" if query is None else str(query).strip()
                if not text or text.lower() == "none":
                    continue
                doc = page_to_idx.get(str(page))
                if doc is not None:
                    queries.append(text)
                    golds.append(doc)
    return queries, golds


def run_paired(task, processor, model, device, ks, embed_bs, num_workers, max_queries, max_docs):
    """ViDoRe V1 / JinaVDR all-page candidate protocol.

    Empty-query rows do not become evaluation queries, but their unique pages
    remain retrieval candidates. Duplicate page images are encoded only once.
    """
    doc_emb, metas = embed_image_stream(
        processor, model, task["paired"], device, embed_bs, num_workers,
        meta_keys=["image_filename"], max_docs=max_docs,
        dedupe_key="image_filename",
    )
    if not metas:
        raise ValueError(f"no candidate pages for paired subset {task['subset']}")

    page_to_idx = {}
    for i, m in enumerate(metas):
        key = m.get("image_filename")
        if key is None:
            raise ValueError(
                f"paired candidate missing image_filename in {task['subset']}"
            )
        key = str(key)
        if key in page_to_idx:
            raise ValueError(f"duplicate page was encoded twice: {key}")
        page_to_idx[key] = i

    queries, golds = _paired_query_rows(task["paired"], page_to_idx)
    if not queries:
        raise ValueError(f"no non-empty queries for paired subset {task['subset']}")
    if max_queries and max_queries < len(queries):
        queries = queries[:max_queries]
        golds = golds[:max_queries]
    q_emb = embed_queries(processor, model, queries, device, embed_bs)
    scores = chunked_maxsim(q_emb, doc_emb)
    relevant = [{g} for g in golds]
    graded = [{g: 1.0} for g in golds]
    if scores.shape[0] <= 16 and scores.shape[1] <= 16:
        idx_to_page = {i: k for k, i in page_to_idx.items()}
        print(f"[rank] {task_tag(task)}")
        for qi, query in enumerate(queries):
            order = scores[qi].argsort(descending=True).tolist()
            gold_page = idx_to_page[golds[qi]]
            hit = "HIT" if order[0] == golds[qi] else "MISS"
            top = [
                f"{idx_to_page[j]}={float(scores[qi, j]):.3f}" for j in order[:3]
            ]
            print(f"[rank] {hit} gold={gold_page}  {query}")
            print(f"[rank]   top {', '.join(top)}")
    return (
        metrics_from_scores(scores, relevant, graded, ks),
        dict(n_docs=doc_emb.shape[0], n_queries=q_emb.shape[0]),
    )


def run_beir(
    task,
    processor,
    model,
    device,
    ks,
    embed_bs,
    num_workers,
    max_queries,
    max_docs,
    corpus_cache=None,
):
    corpus_key = (tuple(task["corpus"]), int(max_docs))
    if corpus_cache is not None and corpus_cache.get("key") == corpus_key:
        doc_emb, metas = corpus_cache["value"]
        print(f"[eval] reuse encoded corpus for {task_tag(task)} ({doc_emb.shape[0]} docs)")
    else:
        doc_emb, metas = embed_image_stream(
            processor, model, task["corpus"], device, embed_bs, num_workers,
            meta_keys=["corpus-id", "id"], max_docs=max_docs,
        )
        if corpus_cache is not None:
            corpus_cache.clear()
            corpus_cache.update(key=corpus_key, value=(doc_emb, metas))
    def _first(m, *keys, default):
        for k in keys:
            v = m.get(k)
            if v is not None:
                return v
        return default

    corpus_ids = [str(_first(m, "corpus-id", "id", default=f"c{i}")) for i, m in enumerate(metas)]
    cid_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    if len(cid_to_idx) != len(corpus_ids):
        raise ValueError(f"duplicate corpus ids in {task['subset']}")

    q_rows = read_parquet_rows(task["queries"])
    q_texts = [str(r["text"] if "text" in r else r["query"]) for r in q_rows]
    q_ids = [str(r.get("id", r.get("query-id", f"q{i}"))) for i, r in enumerate(q_rows)]

    rel_by_q, grad_by_q = defaultdict(set), defaultdict(dict)
    for r in read_parquet_rows(task["qrels"]):
        qid = str(r.get("query-id", r.get("id", "")))
        cid = str(r.get("corpus-id", r.get("id", "")))
        s = float(r.get("score", 1.0) or 1.0)
        if s > 0 and cid in cid_to_idx:
            rel_by_q[qid].add(cid_to_idx[cid])
            grad_by_q[qid][cid_to_idx[cid]] = s

    keep = [
        i for i in range(len(q_ids))
        if q_texts[i].strip() and q_texts[i].lower() != "none"
        and q_ids[i] in rel_by_q and rel_by_q[q_ids[i]]
    ]
    if max_queries and max_queries < len(keep):
        keep = keep[:max_queries]
    if not keep:
        raise ValueError(f"no non-empty queries with valid qrels for {task['subset']}")
    sub_emb = embed_queries(processor, model, [q_texts[i] for i in keep], device, embed_bs)
    sub_ids = [q_ids[i] for i in keep]
    relevant = [rel_by_q[q] for q in sub_ids]
    graded = [grad_by_q[q] for q in sub_ids]

    scores = chunked_maxsim(sub_emb, doc_emb)
    return (
        metrics_from_scores(scores, relevant, graded, ks),
        dict(n_docs=doc_emb.shape[0], n_queries=len(sub_ids)),
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", default="")
    ap.add_argument("--adapter-dir", default=None)
    ap.add_argument(
        "--skip-adapter-weights",
        action="store_true",
        help="keep adapter-dir for run_config/provenance but load only --base-model",
    )
    ap.add_argument("--eval-root", default="")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-visual-tokens", type=int, default=None)
    ap.add_argument("--embed-batch", type=int, default=16)
    ap.add_argument("--ks", default="1,5,10")
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--datasets", default="")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--run-name", default="eval")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--bidirectional-attention", choices=("auto", "on", "off"), default="auto")
    ap.add_argument(
        "--head-dim",
        type=int,
        default=None,
        help=" Matryoshka: which head to score with. Required when the run trained "
        "several heads, since a multi-head forward returns a dict, not one tensor. "
        "Must be one of the checkpoint head_dims.",
    )
    ap.add_argument("--allow-config-override", action="store_true")
    ap.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Archive existing eval dir and start clean (ignores --resume).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Keep good tasks under eval/tasks/; only run missing ones.",
    )
    return ap.parse_args()


def resolve_eval_contract(args):
    """Resolve and validate evaluation settings against the training manifest."""
    config = {}
    config_path = None
    if args.adapter_dir:
        candidate = Path(args.adapter_dir) / "run_config.json"
        if candidate.is_file():
            config_path = candidate
            config = json.loads(candidate.read_text(encoding="utf-8"))

    model_dir = Path(args.adapter_dir or args.base_model or ".")
    model_cfg = _read_json(model_dir / "config.json")
    if not config.get("head_dims") and model_cfg.get("head_dims"):
        config["head_dims"] = model_cfg["head_dims"]

    trained_mvt = config.get("max_visual_tokens")
    trained_bidir = config.get("bidirectional_attention")
    if args.max_visual_tokens is None:
        args.max_visual_tokens = int(trained_mvt) if trained_mvt is not None else 1024
    elif trained_mvt is not None and int(args.max_visual_tokens) != int(trained_mvt):
        if not args.allow_config_override:
            raise ValueError(
                f"MVT mismatch: eval={args.max_visual_tokens}, trained={trained_mvt}; "
                "pass --allow-config-override to force"
            )

    if args.bidirectional_attention == "auto":
        args.bidirectional_attention = (
            trained_bidir if trained_bidir in ("on", "off") else "on"
        )
    elif trained_bidir in ("on", "off") and args.bidirectional_attention != trained_bidir:
        if not args.allow_config_override:
            raise ValueError(
                f"attention mismatch: eval={args.bidirectional_attention}, "
                f"trained={trained_bidir}; pass --allow-config-override to force"
            )

    # guess: refuse rather than silently evaluate an arbitrary width.
    trained_heads = config.get("head_dims")
    if trained_heads:
        if args.head_dim is None:
            raise ValueError(
                f"this run trained heads {trained_heads}; pass --head-dim to choose one "
                "(eval_run.sh loops over all of them)"
            )
        if int(args.head_dim) not in [int(d) for d in trained_heads]:
            raise ValueError(
                f"--head-dim {args.head_dim} is not among the trained heads {trained_heads}"
            )
    elif args.head_dim is not None:
        raise ValueError(
            "--head-dim given but this run is single-head "
            f"(col_dim={config.get('col_dim') or model_cfg.get('dim')}); drop --head-dim"
        )
    return config_path, config


def _read_json(path):
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _file_identity(path, hash_content=False):
    p = Path(path)
    stat = p.stat()
    rec = {
        "path": str(p.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if hash_content:
        digest = hashlib.sha256()
        with p.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        rec["sha256"] = digest.hexdigest()
    return rec


def build_eval_contract(args, train_config, tasks, ks):
    """Fingerprint model, data and scoring settings used by resumable tasks."""
    model_files = []
    for root in (args.base_model, args.adapter_dir):
        if not root:
            continue
        directory = Path(root)
        if not directory.is_dir():
            continue
        candidates = set(directory.glob("*.json"))
        candidates.update(directory.glob("*.safetensors"))
        for path in sorted(candidates):
            hash_content = (
                path.stat().st_size <= 16 * 1024 * 1024
                or path.name == "adapter_model.safetensors"
            )
            model_files.append(_file_identity(path, hash_content=hash_content))

    data_paths = set()
    for task in tasks:
        for field in ("paired", "corpus", "queries", "qrels"):
            data_paths.update(task.get(field) or [])
    data_files = [_file_identity(path) for path in sorted(data_paths)]

    payload = {
        "protocol": EVAL_PROTOCOL,
        "run_name": args.run_name,
        "base_model": str(Path(args.base_model).resolve()),
        "adapter_dir": (
            str(Path(args.adapter_dir).resolve()) if args.adapter_dir else None
        ),
        "skip_adapter_weights": args.skip_adapter_weights,
        "model_files": model_files,
        "training": {
            "col_dim": train_config.get("col_dim"),
            "head_dims": train_config.get("head_dims"),
            "teacher_md5": train_config.get("teacher_md5"),
            "max_visual_tokens": train_config.get("max_visual_tokens"),
            "bidirectional_attention": train_config.get("bidirectional_attention"),
            "attn": train_config.get("attn"),
        },
        "evaluation": {
            "max_visual_tokens": args.max_visual_tokens,
            "bidirectional_attention": args.bidirectional_attention,
            # Part of the contract, not cosmetic: without it the six per-head passes
            # of one run would share a contract id and resume into each other's cache.
            "head_dim": args.head_dim,
            "attn": args.attn,
            "ks": list(ks),
            "max_queries": args.max_queries,
            "max_docs": args.max_docs,
            "datasets": args.datasets,
            "eval_root": str(Path(args.eval_root).resolve()),
        },
        "data_files": data_files,
        "tasks": [task_tag(task) for task in tasks],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def collect_run_metadata(args, train_config, model, world):
    """Provenance for the results table: model / data / training / eval settings."""
    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else None
    adapter_cfg = (
        _read_json(adapter_dir / "adapter_config.json")
        if adapter_dir and not args.skip_adapter_weights
        else {}
    )
    base_cfg = _read_json(Path(args.base_model) / "config.json")
    data_root = train_config.get("data_root")
    export = _read_json(Path(data_root) / "export_stats.json") if data_root else {}

    steps = train_config.get("steps")
    epoch_done = None
    world_train = train_config.get("world_size")
    if adapter_dir:
        ckpts = sorted(
            adapter_dir.glob("checkpoint-*"), key=lambda p: int(p.name.rsplit("-", 1)[-1])
        )
        if ckpts:
            state = _read_json(ckpts[-1] / "trainer_state.json")
            steps = steps or state.get("global_step")
            epoch_done = state.get("epoch")
            world_train = world_train or len(list(ckpts[-1].glob("rng_state_*.pth"))) or None

    per_device = train_config.get("per_device_batch_size")
    accum = train_config.get("grad_accum")
    eff_batch = (
        per_device * accum * world_train
        if None not in (per_device, accum, world_train)
        else None
    )
    total_params = sum(p.numel() for p in model.parameters())
    lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    text_cfg = base_cfg.get("text_config") or {}
    hardneg_loss = train_config.get("hardneg_loss", "negative_ce")
    listwise = hardneg_loss == "listwise"
    hardneg = bool(train_config.get("use_hardnegatives")) or listwise
    git_hash = ""
    if adapter_dir and (adapter_dir / "git_hash.txt").is_file():
        git_hash = (adapter_dir / "git_hash.txt").read_text(encoding="utf-8").strip()

    return dict(
        model=dict(
            run_name=args.run_name,
            adapter_dir=str(adapter_dir) if adapter_dir else None,
            base_model=train_config.get("base_model") or args.base_model,
            experiment_type="/".join(
                [
                    "EVIE" if args.skip_adapter_weights else ("LoRA" if adapter_cfg else "full-ft"),
                    "bidir" if args.bidirectional_attention == "on" else "causal",
                    "listwise" if listwise else ("hardneg" if hardneg else "in-batch"),
                ]
            ),
            total_params=total_params,
            trainable_params=lora_params,
            trainable_ratio=round(lora_params / total_params, 6) if total_params else None,
            # head's, not the warm start's config.json dim (Preview says 128).
            emb_dim=args.head_dim if args.head_dim is not None else base_cfg.get("dim"),
            head_dim=args.head_dim,
            head_dims=train_config.get("head_dims"),
            teacher_dir=train_config.get("teacher_dir"),
            teacher_md5=train_config.get("teacher_md5"),
            hidden_size=text_cfg.get("hidden_size"),
            num_hidden_layers=text_cfg.get("num_hidden_layers"),
            dtype="bfloat16",
            attn=args.attn,
            git_hash=git_hash,
        ),
        data=dict(
            data_root=data_root,
            version=export.get("version"),
            policy=export.get("policy"),
            train_pairs=export.get("written_rows"),
            by_source=export.get("by_source"),
            max_samples_per_source=train_config.get("max_samples_per_source"),
            hardneg_root=(
                train_config.get("listwise_root") if listwise else train_config.get("hardneg_root")
            )
            if hardneg
            else None,
            num_hard_negs=train_config.get("num_hard_negs") if not listwise and hardneg else None,
        ),
        training=dict(
            seed=train_config.get("seed"),
            epochs=train_config.get("epochs"),
            epoch_done=epoch_done,
            steps=steps,
            world_size=world_train,
            per_device_batch_size=per_device,
            grad_accum=accum,
            effective_batch_size=eff_batch,
            consumed_samples=steps * eff_batch if (steps and eff_batch) else None,
            learning_rate=train_config.get("learning_rate"),
            lr_scheduler="cosine",
            warmup_ratio=train_config.get("warmup_ratio"),
            weight_decay=train_config.get("weight_decay"),
            lora_r=adapter_cfg.get("r", train_config.get("lora_r")),
            lora_alpha=adapter_cfg.get("lora_alpha", train_config.get("lora_alpha")),
            lora_dropout=adapter_cfg.get("lora_dropout", train_config.get("lora_dropout")),
            lora_target_modules=adapter_cfg.get("target_modules"),
            loss=(
                "ColbertListwiseKLLoss"
                if listwise
                else ("ColbertNegativeCELoss" if hardneg else "ColbertLoss")
            ),
            loss_temperature=train_config.get("loss_temperature"),
            in_batch_term_weight=train_config.get("hardneg_in_batch_weight") if hardneg else None,
            max_visual_tokens=train_config.get("max_visual_tokens"),
            grad_checkpointing=train_config.get("grad_checkpointing"),
            train_runtime_seconds=train_config.get("train_runtime_seconds"),
        ),
        eval=dict(
            eval_root=args.eval_root,
            eval_protocol=EVAL_PROTOCOL,
            max_visual_tokens=args.max_visual_tokens,
            bidirectional_attention=args.bidirectional_attention,
            embed_batch=args.embed_batch,
            num_workers=args.num_workers,
            world_size=world,
            datasets_filter=args.datasets or None,
            max_queries=args.max_queries or None,
            max_docs=args.max_docs or None,
            truncated=bool(args.max_queries or args.max_docs),
            date=time.strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )


DS_LABEL = {
    "eval": "ViDoRe V1",
    "eval_v2": "ViDoRe V2",
    "eval_v3": "ViDoRe V3",
    "JinaVDR": "JinaVDR",
    "demo": "Bundled demo",
}


def write_summary_md(path, summary, ks):
    md = summary["metadata"]
    model, data, train, ev = md["model"], md["data"], md["training"], md["eval"]
    head, avg, per_ds = summary["headline"], summary["averages"], summary["per_dataset"]

    def pct(v):
        return "—" if v is None else f"{v * 100:.2f}"

    def num(v):
        return "—" if v is None else f"{v:,}"

    def params(v):
        return "—" if v is None else (f"{v / 1e9:.2f}B" if v >= 1e9 else f"{v / 1e6:.1f}M")

    lines = [
        f"# {model['run_name']} eval",
        "",
        f"status `{summary['status']}` · {summary['completed_tasks']}/{summary['expected_tasks']} complete · "
        f"{summary['n_failed']} failed · {ev['date']}",
        "",
        "## Headline",
        "",
        "| model | type | size | trainable | emb | train pairs | "
        "V1 nDCG@10 | V2 nDCG@10 | V3 nDCG@10 | Jina nDCG@10 | avg4 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {model['run_name']} | {model['experiment_type']} | {params(model['total_params'])} | "
        f"{params(model['trainable_params'])} | {model['emb_dim']} | {num(data['train_pairs'])} | "
        f"{pct(head.get('V1 nDCG@10'))} | {pct(head.get('V2 nDCG@10'))} | "
        f"{pct(head.get('V3 nDCG@10'))} | {pct(head.get('JinaVDR nDCG@10'))} | "
        f"{pct(avg.get('avg4_ndcg@10'))} |",
        "",
        "## Metrics (four families × @1/@5/@10)",
        "",
        "| board | subsets | queries | docs | " + " | ".join(f"{m}@{k}" for m in METRIC_NAMES for k in ks) + " |",
        "| --- | ---: | ---: | ---: |" + " ---: |" * (len(METRIC_NAMES) * len(ks)),
    ]
    for ds, label in DS_LABEL.items():
        e = per_ds.get(ds)
        if not e:
            continue
        cells = " | ".join(pct(e.get(f"{m}@{k}")) for m in METRIC_NAMES for k in ks)
        lines.append(
            f"| {label} | {e['n_subsets']}/{e['n_subsets_expected']} | "
            f"{num(e['n_queries'])} | {num(e['n_docs'])} | {cells} |"
        )

    lines += [
        "",
        "## Config",
        "",
        "| key | value |",
        "| --- | --- |",
        f"| base_model | `{model['base_model']}` |",
        f"| architecture | hidden {model['hidden_size']} × {model['num_hidden_layers']} · emb {model['emb_dim']} |",
        f"| dtype / attn | {model['dtype']} / {model['attn']} |",
        f"| data | {data['version'] or '—'} · {data['policy'] or '—'} · {num(data['train_pairs'])} pairs |",
        f"| data_root | `{data['data_root']}` |",
        f"| hardneg | {data['hardneg_root'] or '—'} · num_negs={data['num_hard_negs'] or '—'} |",
        f"| seed / epochs / steps | {train['seed']} / {train['epochs']} / {train['steps']} |",
        f"| batch | per_device {train['per_device_batch_size']} × accum {train['grad_accum']} "
        f"× world {train['world_size']} = {train['effective_batch_size']} |",
        f"| consumed samples | {num(train['consumed_samples'])} |",
        f"| lr / warmup / wd | {train['learning_rate']} ({train['lr_scheduler']}) / "
        f"{train['warmup_ratio']} / {train['weight_decay']} |",
        f"| LoRA | r={train['lora_r']} α={train['lora_alpha']} dropout={train['lora_dropout']} |",
        f"| loss | {train['loss']} · T={train['loss_temperature'] or '—'} · "
        f"in_batch_weight={train['in_batch_term_weight'] or '—'} |",
        f"| MVT train / eval | {train['max_visual_tokens']} / {ev['max_visual_tokens']} |",
        f"| bidir eval | {ev['bidirectional_attention']} |",
        f"| protocol | `{ev.get('eval_protocol') or '—'}` |",
        f"| eval data | `{ev['eval_root']}` · ks={ks} · truncated={ev['truncated']} |",
    ]
    if summary["failed_tasks"]:
        lines += ["", "## Failed tasks", ""]
        lines += [f"- `{r['task']}`: {r['error']}" for r in summary["failed_tasks"]]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _task_corpus_size(task):
    """Approx embed workload (no image decode) for load balancing."""
    paths = task.get("paired") or task.get("corpus") or []
    if not paths:
        return 0
    try:
        return sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
    except Exception:
        return len(paths)


def _balanced_shard(tasks, world, rank):
    """Greedy split while keeping tasks that share a corpus on the same rank."""
    grouped = {}
    for task in tasks:
        key = (
            ("beir-corpus", tuple(task["corpus"]))
            if task["fmt"] == "beir"
            else ("single-task", task_tag(task))
        )
        grouped.setdefault(key, []).append(task)
    sized = sorted(
        (
            (max(_task_corpus_size(task) for task in group), i, group)
            for i, group in enumerate(grouped.values())
        ),
        key=lambda item: -item[0],
    )
    loads = [0] * world
    buckets = [[] for _ in range(world)]
    for sz, _, group in sized:
        r = min(range(world), key=lambda w: loads[w])
        buckets[r].extend(group)
        loads[r] += sz
    return buckets[rank]


def setup_ddp():
    if "LOCAL_RANK" not in os.environ and "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", rank))
    timeout = timedelta(seconds=int(os.environ.get("EVAL_DDP_TIMEOUT_S", "21600")))
    # Bind this process to its GPU before the first collective, otherwise every
    # rank allocates NCCL buffers on cuda:0 ("Duplicate GPU detected").
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        torch.distributed.init_process_group(
            backend="nccl", device_id=torch.device(f"cuda:{local}"), timeout=timeout
        )
    else:
        torch.distributed.init_process_group(backend="gloo", timeout=timeout)
    return rank, world, local


def main():
    args = parse_args()
    config_path, train_config = resolve_eval_contract(args)
    rank, world, local = setup_ddp()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    if tuple(ks) != REQUIRED_KS:
        raise ValueError(
            f"complete evaluation requires --ks 1,5,10; got {args.ks!r}"
        )
    out_dir = forbid_venv_path(args.output_dir, "output-dir")
    resume = bool(args.resume) and not args.overwrite_output

    tasks = discover_tasks(args.eval_root)
    if args.datasets:
        want = {d.strip() for d in args.datasets.split(",") if d.strip()}
        tasks = [t for t in tasks if t["dataset"] in want]
    else:
        tasks = [t for t in tasks if t["dataset"] != "demo"]
    expected_tasks = len(tasks)
    expected_by_ds = Counter(t["dataset"] for t in tasks)
    if not args.datasets:
        want_full = {"eval": 10, "eval_v2": 4, "eval_v3": 48, "JinaVDR": 76}
        bad = {
            key: (want_full[key], expected_by_ds.get(key, 0))
            for key in want_full
            if expected_by_ds.get(key, 0) != want_full[key]
        }
        if bad:
            raise RuntimeError(f"eval discovery incomplete: {bad}; refuse to start")

    # Only rank 0 mutates the shared output path.
    prep_error = [None]
    cached_payload = [None]
    contract_payload = [None]
    if rank == 0:
        try:
            contract_payload[0] = build_eval_contract(args, train_config, tasks, ks)
            if args.overwrite_output and out_dir.exists() and any(out_dir.iterdir()):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                archive = out_dir.with_name(f"{out_dir.name}_archive_{stamp}")
                out_dir.rename(archive)
                print(f"[eval] archived previous output -> {archive}")
            elif out_dir.exists() and any(out_dir.iterdir()) and not resume:
                raise FileExistsError(
                    f"non-empty eval output: {out_dir}; pass --resume to continue "
                    "or --overwrite-output to archive it"
                )
            out_dir.mkdir(parents=True, exist_ok=True)
            # Drop stale rank shards from prior world-size / crashed runs so merge
            # never re-ingests old errors or metrics from leftover rank_*.json.
            for stale in out_dir.glob("rank_*.json"):
                stale.unlink()
            for stale in (out_dir / "summary.json", out_dir / "summary.md"):
                stale.unlink(missing_ok=True)
            cached_payload[0] = load_cached_task_map(
                out_dir, contract_id=contract_payload[0]
            )
            print(
                f"[eval] run={args.run_name} protocol={EVAL_PROTOCOL} "
                f"contract={contract_payload[0][:12]} "
                f"mode={'resume' if resume else 'fresh'} "
                f"cached_ok={len(cached_payload[0])} out={out_dir}"
            )
        except Exception as exc:
            prep_error[0] = repr(exc)
    if world > 1:
        torch.distributed.broadcast_object_list(prep_error, src=0)
        torch.distributed.broadcast_object_list(cached_payload, src=0)
        torch.distributed.broadcast_object_list(contract_payload, src=0)
    if prep_error[0]:
        if world > 1:
            torch.distributed.destroy_process_group()
        raise RuntimeError(f"eval output preflight failed: {prep_error[0]}")
    if world > 1:
        torch.distributed.barrier()
    cached = cached_payload[0] or {}
    eval_contract_id = contract_payload[0]

    model_source = args.base_model
    is_adapter = bool(
        args.adapter_dir
        and not args.skip_adapter_weights
        and (Path(args.adapter_dir) / "adapter_config.json").is_file()
    )
    if args.adapter_dir and not is_adapter and not args.skip_adapter_weights:
        model_source = args.adapter_dir
    print(
        f"[eval rank{rank}] loading model on {device} "
        f"(source={model_source}, adapter={args.adapter_dir if is_adapter else None})"
    )
    processor = ColQwen3_5Processor.from_pretrained(
        model_source, max_num_visual_tokens=args.max_visual_tokens
    )
    model_config = Qwen3_5Config.from_pretrained(model_source)
    trained_col_dim = train_config.get("col_dim")
    trained_heads = train_config.get("head_dims") or getattr(
        model_config, "head_dims", None
    )
    if trained_heads:
        model_config.head_dims = [int(d) for d in trained_heads]
        model_config.dim = max(model_config.head_dims)
        mrl_prefix = bool(train_config.get("mrl_prefix", getattr(model_config, "mrl_prefix", False)))
        model_config.mrl_prefix = mrl_prefix
        layout = "prefix-MRL" if mrl_prefix else "ModuleDict"
        print(
            f"[eval rank{rank}] custom_text_proj = {layout}{model_config.head_dims}, "
            f"scoring head d={args.head_dim}"
        )
    elif is_adapter:
        if trained_col_dim is None:
            raise ValueError("adapter evaluation requires col_dim in run_config.json")
        model_config.dim = int(trained_col_dim)
        print(f"[eval rank{rank}] custom_text_proj dim={model_config.dim} from run_config.json")
    model = ColQwen3_5.from_pretrained(
        model_source,
        config=model_config,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn,
    )
    if args.bidirectional_attention == "on":
        model.enable_bidirectional_attention()
        print("[model] bidirectional attention ON")
    else:
        print("[model] bidirectional attention OFF (causal)")
    if is_adapter:
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    if trained_heads:
        # After PEFT wrapping, so the selection reaches the ColQwen3_5 underneath.
        # Every downstream scorer expects one tensor per forward.
        set_active_head(model, int(args.head_dim))
    model = model.to(device).eval()

    pending = [
        t for t in tasks
        if not (resume and task_tag(t) in cached)
    ]

    if world > 1:
        pending = _balanced_shard(pending, world, rank)
    print(
        f"[eval rank{rank}] pending {len(pending)} "
        f"(expected_total={expected_tasks}, cached_ok={len(cached)}) "
        f"(~{sum(_task_corpus_size(t) for t in pending)} corpus imgs) "
        f"counts={dict(expected_by_ds)}"
    )

    results = []
    # One-entry GPU cache. _balanced_shard keeps shared V3 language tasks
    # contiguous on the same rank, so the corpus tensor is reused six times
    # without accumulating multiple domains in VRAM.
    beir_corpus_cache = {}
    for t in pending:
        tag = task_tag(t)
        try:
            t0 = time.time()
            if t["fmt"] == "paired":
                metrics, info = run_paired(
                    t, processor, model, device, ks, args.embed_batch,
                    args.num_workers, args.max_queries, args.max_docs,
                )
            else:
                metrics, info = run_beir(
                    t, processor, model, device, ks, args.embed_batch,
                    args.num_workers, args.max_queries, args.max_docs,
                    corpus_cache=beir_corpus_cache,
                )
            dt = time.time() - t0
            rec = dict(
                task=tag, dataset=t["dataset"], subset=t["subset"], lang=t.get("lang"),
                fmt=t["fmt"], metrics=metrics, info=info, seconds=round(dt, 1),
                eval_protocol=EVAL_PROTOCOL,
                eval_contract_id=eval_contract_id,
                run_name=args.run_name,
            )
            results.append(rec)
            save_task_result(
                out_dir,
                rec,
                run_name=args.run_name,
                contract_id=eval_contract_id,
            )
            print(
                f"[eval rank{rank}] {tag}: {metrics}  "
                f"({info['n_queries']}q / {info['n_docs']}d, {dt:.1f}s)"
            )
        except Exception as e:
            import traceback
            print(f"[eval rank{rank}] {tag} FAILED: {e!r}")
            traceback.print_exc()
            results.append(
                dict(
                    task=tag, dataset=t["dataset"], subset=t["subset"], lang=t.get("lang"),
                    fmt=t["fmt"], error=str(e), eval_protocol=EVAL_PROTOCOL,
                    eval_contract_id=eval_contract_id,
                    run_name=args.run_name,
                )
            )

    rank_path = out_dir / f"rank_{rank}.json"
    rank_tmp = out_dir / f".rank_{rank}.json.tmp.{os.getpid()}"
    rank_tmp.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rank_tmp.replace(rank_path)
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"[eval rank{rank}] peak VRAM: {peak:.2f} GB on {device}")

    if world > 1:
        # Evaluation ranks have highly uneven runtimes (large V3 corpora versus
        # tiny paired sets). Do not hold finished ranks in a NCCL collective:
        # the old final barrier deterministically timed out while slow ranks
        # were still encoding. Rank 0 finalizes through atomic shared-FS shards.
        torch.distributed.destroy_process_group()
    if rank != 0:
        return

    finalize_timeout = int(os.environ.get("EVAL_FINALIZE_TIMEOUT_S", "21600"))
    deadline = time.monotonic() + finalize_timeout
    last_report = 0.0
    missing_ranks = list(range(world))
    while missing_ranks:
        missing_ranks = []
        for r in range(world):
            p = out_dir / f"rank_{r}.json"
            try:
                json.loads(p.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                missing_ranks.append(r)
        if not missing_ranks:
            break
        now = time.monotonic()
        if now >= deadline:
            print(
                f"[eval] finalize timeout after {finalize_timeout}s; "
                f"missing rank shards: {missing_ranks}"
            )
            break
        if now - last_report >= 60:
            print(
                f"[eval] waiting for {len(missing_ranks)}/{world} rank shards: "
                f"{missing_ranks}"
            )
            last_report = now
        time.sleep(2)

    incomplete = [0]
    if rank == 0:
        # Prefer durable tasks/ cache; fold in this-run failures from rank_*.json.
        # Ignore leftover rank_N.json from a previous larger world size.
        merged_map = load_cached_task_map(
            out_dir, contract_id=eval_contract_id
        )
        failed = []
        for r in range(world):
            p = out_dir / f"rank_{r}.json"
            if not p.exists():
                continue
            for rec in json.loads(p.read_text(encoding="utf-8")):
                if "error" in rec:
                    failed.append(rec)
                elif _is_ok_task_rec(rec, contract_id=eval_contract_id):
                    merged_map[rec["task"]] = rec
                # Ignore unrecognized rows (e.g. protocol-mismatched leftovers).
        # Drop failures that were successfully produced this session (or still cached).
        failed = [f for f in failed if f["task"] not in merged_map]
        order = {task_tag(t): i for i, t in enumerate(tasks)}
        expected_tags = set(order)
        missing_tags = [task_tag(t) for t in tasks if task_tag(t) not in merged_map]
        merged = sorted(
            (rec for rec in merged_map.values() if rec["task"] in expected_tags),
            key=lambda rec: order.get(rec["task"], 10**9),
        )

        failed_count = len(failed)
        by_ds = defaultdict(list)
        for rec in merged:
            by_ds[rec["dataset"]].append(rec)

        def avg_metric(recs, name):
            vals = [r["metrics"].get(name) for r in recs]
            vals = [v for v in vals if v is not None]
            return (sum(vals) / len(vals)) if vals else None

        per_ds = {}
        for ds, recs in by_ds.items():
            entry = {f"{m}@{k}": avg_metric(recs, f"{m}@{k}") for m in METRIC_NAMES for k in ks}
            entry["n_subsets"] = len(recs)
            entry["n_subsets_expected"] = expected_by_ds.get(ds, len(recs))
            entry["n_queries"] = sum(r["info"]["n_queries"] for r in recs)
            entry["n_docs"] = sum(r["info"]["n_docs"] for r in recs)
            per_ds[ds] = entry

        hdr = "  ".join(f"{m + '@' + str(k):>9}" for m in METRIC_NAMES for k in ks)

        def fmt_row(get):
            return "  ".join(f"{(get(f'{m}@{k}') or 0) * 100:8.2f}%" for m in METRIC_NAMES for k in ks)

        print("\n================ EVAL SUMMARY ================")
        for ds in DS_LABEL:
            if ds not in by_ds:
                continue
            m = per_ds[ds]
            print(
                f"\n-- {DS_LABEL[ds]} ({m['n_subsets']}/{m['n_subsets_expected']} subsets, "
                f"{m['n_queries']}q / {m['n_docs']}d) --"
            )
            print(f"   {hdr}")
            print(f"   {fmt_row(m.get)}")

        all_recs = [r for ds in by_ds for r in by_ds[ds]]
        print(f"\n-- ALL DATASETS ({len(all_recs)}/{expected_tasks} subsets) --")
        print(f"   {hdr}")
        print(f"   {fmt_row(lambda name: avg_metric(all_recs, name))}")

        headline = {
            "V1 nDCG@10": per_ds.get("eval", {}).get("ndcg@10"),
            "V2 nDCG@10": per_ds.get("eval_v2", {}).get("ndcg@10"),
            "V3 nDCG@10": per_ds.get("eval_v3", {}).get("ndcg@10"),
            "JinaVDR nDCG@10": per_ds.get("JinaVDR", {}).get("ndcg@10"),
        }
        four = [headline[k] for k in (
            "V1 nDCG@10", "V2 nDCG@10", "V3 nDCG@10", "JinaVDR nDCG@10"
        )]
        averages = {
            "avg4_ndcg@10": (
                sum(four) / 4 if all(v is not None for v in four) else None
            ),
        }
        print("\n================ HEADLINE (nDCG@10) ================")
        for name, v in headline.items():
            print(f"   {name:<18}: {'n/a' if v is None else f'{v * 100:.2f}%'}")
        avg4 = averages["avg4_ndcg@10"]
        print(f"   {'Avg4 nDCG@10':<18}: {'n/a' if avg4 is None else f'{avg4 * 100:.2f}%'}")

        invalid_metric_tasks = [
            rec["task"] for rec in merged if not _has_complete_metrics(rec["metrics"])
        ]
        # Complete iff every task belongs to this exact contract and contains all
        # four metric families at @1/@5/@10.
        complete = not failed and not missing_tags and not invalid_metric_tasks
        metadata = collect_run_metadata(args, train_config, model, world)
        summary = dict(
            status="complete" if complete else "partial",
            run_name=args.run_name,
            adapter_dir=args.adapter_dir,
            eval_protocol=EVAL_PROTOCOL,
            eval_contract_id=eval_contract_id,
            training_config=str(config_path) if config_path else None,
            max_visual_tokens=args.max_visual_tokens,
            bidirectional_attention=args.bidirectional_attention,
            ks=ks,
            expected_tasks=expected_tasks,
            completed_tasks=expected_tasks - len(missing_tags),
            n_failed=failed_count,
            n_cached=len(cached),
            n_ran=sum(1 for r in results if "metrics" in r) if world == 1 else None,
            failed_tasks=failed,
            missing_tasks=missing_tags,
            invalid_metric_tasks=invalid_metric_tasks,
            headline=headline,
            averages=averages,
            metadata=metadata,
            per_dataset=per_ds,
            per_task=merged,
        )
        summary_path = out_dir / "summary.json"
        summary_tmp = out_dir / f".summary.json.tmp.{os.getpid()}"
        summary_tmp.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary_tmp.replace(summary_path)
        write_summary_md(out_dir / "summary.md", summary, ks)
        print(f"\n[eval] summary written to {out_dir / 'summary.json'} and summary.md")
        if not complete:
            print(
                f"\n[eval] INCOMPLETE: {len(merged)}/{expected_tasks} ok, "
                f"{failed_count} failed this run, {len(missing_tags)} missing"
            )
            for rec in failed:
                print(f"  - FAIL {rec['task']}: {rec['error']}")
            for tag in missing_tags[:20]:
                print(f"  - MISS {tag}")
            if len(missing_tags) > 20:
                print(f"  - ... {len(missing_tags) - 20} more missing")
            print(
                f"[eval] tip: delete bad files under {tasks_dir(out_dir)} then "
                "re-run with --resume to patch only those tasks"
            )
        incomplete[0] = 0 if complete else 1
    if incomplete[0]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
