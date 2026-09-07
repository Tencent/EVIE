"""Load `part_*.parquet` query/image pairs (and optional hard-negative shards)."""

from __future__ import annotations

import hashlib
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import datasets
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import Dataset
from datasets.arrow_writer import ArrowWriter

from colpali_engine.data.dataset import ColPaliEngineDataset
from paths import forbid_venv_path

ALLOWED_SOURCES = (
    "colpali_train_set",
    "VisRAG-Ret-Train-Synthetic-data",
    "VisRAG-Ret-Train-In-domain-data",
    "vdr-multilingual-train",
    "tatdqa_train",
    "tabfquad_train_set",
)
_STREAM_BATCH_ROWS = 256


def _set_task_ids_from_sources(
    dataset: ColPaliEngineDataset,
    sources: pa.Array | pa.ChunkedArray,
) -> None:
    if isinstance(sources, pa.ChunkedArray):
        sources = sources.combine_chunks()
    encoded = pc.dictionary_encode(sources)
    dataset.task_ids = encoded.indices.to_numpy(zero_copy_only=False)
    dataset.task_names = encoded.dictionary.to_pylist()
    counts = pc.value_counts(encoded.indices).to_pylist()
    summary = {
        dataset.task_names[int(item["values"])]: int(item["counts"])
        for item in counts
    }
    print(f"[data] task-consistent groups: {summary}")


def _attach_task_ids(
    dataset: ColPaliEngineDataset,
    queries_ds: Dataset,
    corpus_ds: Dataset,
    docid_to_idx: dict[int, int] | None,
) -> None:
    if "source" not in corpus_ds.column_names:
        dataset.task_ids = [0] * len(dataset)
        dataset.task_names = ["all"]
        return
    positive_column = queries_ds.data.column("pos_target")
    if docid_to_idx is None:
        row_indices = positive_column
    else:
        row_indices = pa.array(
            [docid_to_idx[int(doc_id)] for doc_id in positive_column.to_pylist()],
            type=pa.int64(),
        )
    sources = pc.take(corpus_ds.data.column("source"), row_indices)
    _set_task_ids_from_sources(dataset, sources)


def _default_dataset_cache_dir() -> Path:
    override = os.environ.get("HF_DATASETS_CACHE") or os.environ.get("EVIE_DATASETS_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "huggingface" / "datasets"


def _export_parts(data_root: Path) -> list[str]:
    parts = sorted(Path(data_root).glob("part_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No part_*.parquet under {data_root}")
    return [str(p) for p in parts]


def _cache_dir() -> Path:
    d = Path(os.environ.get("HF_DATASETS_CACHE") or _default_dataset_cache_dir())
    forbid_venv_path(d, "HF_DATASETS_CACHE")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fingerprint(parts: list[str]) -> str:
    h = hashlib.sha1()
    for p in parts:
        st = Path(p).stat()
        h.update(Path(p).name.encode())
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()[:16]


def _prepare_streamed_arrow(parts: list[str]) -> Path:
    """Rank 0 writes a shared Arrow cache; other ranks memory-map it.

    Avoid Dataset.filter here: concurrent DDP ranks would race on the same
    cache-*.arrow files.
    """
    final = _cache_dir() / f"export_stream_{_fingerprint(parts)}.arrow"
    done = Path(str(final) + ".done")
    error = Path(str(final) + ".error")
    rank = int(os.environ.get("RANK", "0"))
    if done.exists() and final.exists():
        return final

    if rank == 0:
        tmp = Path(str(final) + f".{os.getpid()}.tmp")
        error.unlink(missing_ok=True)
        try:
            writer = ArrowWriter(path=str(tmp))
            total = 0
            t0 = time.time()
            for i, p in enumerate(parts):
                for rb in pq.ParquetFile(p).iter_batches(batch_size=_STREAM_BATCH_ROWS):
                    writer.write_table(pa.Table.from_batches([rb]))
                    total += rb.num_rows
                print(f"[data] shard {i + 1}/{len(parts)}: {total:,} rows, {time.time() - t0:.0f}s", flush=True)
            writer.finalize()
            os.replace(tmp, final)
            done.touch()
            print(f"[data] arrow cache {final} ({total:,} rows)", flush=True)
            return final
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            error.write_text(repr(exc), encoding="utf-8")
            raise

    waited = 0
    timeout = int(os.environ.get("EVIE_DATA_CACHE_TIMEOUT", "7200"))
    while not (done.exists() and final.exists()):
        if error.exists():
            raise RuntimeError(f"rank0 failed to build Arrow cache: {error.read_text()}")
        time.sleep(5)
        waited += 5
        if waited >= timeout:
            raise TimeoutError(f"timed out waiting for Arrow cache after {timeout}s")
    return final


def build_train_dataset(
    data_root: str | Path,
    sources: Iterable[str] = (),
    max_samples_per_source: int = 0,
) -> ColPaliEngineDataset:
    root = Path(data_root)
    parts = _export_parts(root)
    arrow = _prepare_streamed_arrow(parts)
    ds = Dataset.from_file(str(arrow))

    selected = tuple(dict.fromkeys(sources))
    if selected and set(selected) != set(ALLOWED_SOURCES):
        keep = set(selected)
        ds = ds.filter(lambda s: s in keep, input_columns="source")

    if max_samples_per_source:
        counts = defaultdict(int)
        indices = []
        for i, source in enumerate(ds["source"]):
            if counts[source] < max_samples_per_source:
                indices.append(i)
                counts[source] += 1
        ds = ds.select(indices)

    ds = ds.cast_column("image", datasets.Image())
    tag = ",".join(selected) if selected else "all"
    print(f"[data] {len(ds):,} pairs from {root.name} ({tag})")
    wrapped = ColPaliEngineDataset(ds, pos_target_column_name="image")
    if "source" in ds.column_names:
        _set_task_ids_from_sources(wrapped, ds.data.column("source"))
    return wrapped


def build_hardneg_dataset(
    hardneg_root: str | Path,
    data_root: str | Path,
    num_negatives: int = 2,
    max_samples: int = 0,
    use_negatives: bool = True,
    queries_subdir: str = "",
) -> ColPaliEngineDataset:
    from datasets import load_from_disk
    from colpali_engine.data.dataset import Corpus

    root = Path(hardneg_root)
    subdir = queries_subdir or os.environ.get("HARDNEG_SUBDIR", "judged")
    queries_ds = load_from_disk(str(root / subdir))
    print(f"[data] hardneg subdir: {subdir}")
    corpus_path = root / "corpus"
    if corpus_path.is_dir():
        corpus_ds = load_from_disk(str(corpus_path))
        corpus_origin = str(corpus_path)
    else:
        parts = _export_parts(Path(data_root))
        arrow = _prepare_streamed_arrow(parts)
        corpus_ds = Dataset.from_file(str(arrow)).cast_column("image", datasets.Image())
        corpus_origin = f"export {Path(data_root).name}"

    if max_samples and len(queries_ds) > max_samples:
        queries_ds = queries_ds.shuffle(seed=42).select(range(max_samples))

    docid_to_idx = None
    if "doc_id" in corpus_ds.column_names:
        docid_to_idx = {int(corpus_ds[i]["doc_id"]): i for i in range(len(corpus_ds))}
    corpus = Corpus(corpus_data=corpus_ds, docid_to_idx_mapping=docid_to_idx, doc_column_name="image")

    rename = {}
    if "positive_doc_id" in queries_ds.column_names:
        rename["positive_doc_id"] = "pos_target"
    if "negative_doc_ids" in queries_ds.column_names:
        rename["negative_doc_ids"] = "neg_target"
    for old, new in rename.items():
        if old in queries_ds.column_names and new not in queries_ds.column_names:
            queries_ds = queries_ds.rename_column(old, new)

    if "pos_target" not in queries_ds.column_names:
        raise ValueError("hardneg queries need positive_doc_id / pos_target")

    min_negs = max(1, int(num_negatives)) if use_negatives else 1
    if "neg_target" in queries_ds.column_names:
        before = len(queries_ds)
        lengths = pc.list_value_length(queries_ds.data.column("neg_target"))
        valid = pc.fill_null(pc.greater_equal(lengths, min_negs), False)
        n_valid = int(pc.sum(pc.cast(valid, pa.int64())).as_py())
        if n_valid != before:
            queries_ds = queries_ds.select(pc.indices_nonzero(valid).to_pylist())
        print(
            f"[data] hardneg {before} -> {len(queries_ds)} (>= {min_negs} negs); "
            f"corpus={len(corpus_ds)} from {corpus_origin}"
        )

    if use_negatives:
        if "neg_target" not in queries_ds.column_names:
            raise ValueError("hardneg queries need negative_doc_ids / neg_target")
        wrapped = ColPaliEngineDataset(
            queries_ds,
            corpus=corpus,
            query_column_name="query",
            pos_target_column_name="pos_target",
            neg_target_column_name="neg_target",
            num_negatives=num_negatives,
        )
        _attach_task_ids(wrapped, queries_ds, corpus_ds, docid_to_idx)
        return wrapped

    print(f"[data] hardneg in-batch only: queries={len(queries_ds)}")
    wrapped = ColPaliEngineDataset(
        queries_ds,
        corpus=corpus,
        query_column_name="query",
        pos_target_column_name="pos_target",
        neg_target_column_name=None,
        num_negatives=0,
    )
    _attach_task_ids(wrapped, queries_ds, corpus_ds, docid_to_idx)
    return wrapped
