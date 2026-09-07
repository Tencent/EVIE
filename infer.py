#!/usr/bin/env python3
"""Score document page images against a text query.

For Matryoshka checkpoints, --head selects one projection (default: widest).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "colpali"))

from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor
from colpali_engine.models.qwen3_5.colqwen3_5.modeling_colqwen3_5 import set_active_head


def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def _default_model() -> str:
    if (ROOT / "model.safetensors").is_file():
        return str(ROOT)
    cfg = ROOT / "config.json"
    if cfg.is_file():
        heads = json.loads(cfg.read_text(encoding="utf-8")).get("head_dims") or []
        if heads:
            return "tencent/EVIE-4.5B"
    return "tencent/EVIE-8B"


def load(model_id: str, device: str = "cuda", head: int | None = None, attn: str = "auto"):
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available; falling back to CPU", file=sys.stderr)
        device = "cpu"

    dtype = torch.bfloat16 if "cuda" in device else torch.float32
    if attn == "auto":
        attn_impl = "flash_attention_2" if ("cuda" in device and _has_flash_attn()) else "sdpa"
    else:
        attn_impl = attn

    model = ColQwen3_5.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation=attn_impl,
    ).eval()
    model.enable_bidirectional_attention()
    if getattr(model, "head_dims", None):
        set_active_head(model, head if head is not None else max(model.head_dims))
    elif head is not None:
        set_active_head(model, head)
    return model, ColQwen3_5Processor.from_pretrained(model_id)


@torch.inference_mode()
def score(model, processor, images, queries):
    image_embeddings = model(**processor.process_images(images).to(model.device))
    model.rope_deltas = None
    query_embeddings = model(**processor.process_queries(queries).to(model.device))
    return processor.score(query_embeddings, image_embeddings)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=_default_model())
    ap.add_argument("--head", type=int, default=None, help="Matryoshka head width, e.g. 64/128/2048.")
    ap.add_argument("--query", required=True, action="append")
    ap.add_argument("--image", required=True, action="append")
    ap.add_argument("--device", default="cuda", help="Target device (default: cuda, with fallback to cpu).")
    ap.add_argument(
        "--attn",
        default="auto",
        choices=("auto", "flash_attention_2", "sdpa", "eager"),
        help="Attention implementation (default: auto; flash_attention_2 if available else sdpa).",
    )
    args = ap.parse_args()

    for p in args.image:
        if not Path(p).is_file():
            raise FileNotFoundError(f"Image not found: {p}")

    model, processor = load(args.model, args.device, args.head, args.attn)
    images = [Image.open(p).convert("RGB") for p in args.image]
    scores = score(model, processor, images, args.query)
    for query, row in zip(args.query, scores):
        ranked = sorted(zip(args.image, row.tolist()), key=lambda x: -x[1])
        print(query)
        for path, value in ranked:
            print("  %8.3f  %s" % (value, path))


if __name__ == "__main__":
    main()
