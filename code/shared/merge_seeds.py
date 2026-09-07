#!/usr/bin/env python3
"""Average full state dicts of LoRA runs after merge_and_unload."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers.models.qwen3_5 import Qwen3_5Config

from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor
from paths import forbid_venv_path

CONTRACT = (
    "base_model",
    "col_dim",
    "max_visual_tokens",
    "bidirectional_attention",
    "attn",
    "grad_checkpointing",
    "framework",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "use_hardnegatives",
    "loss_temperature",
    "hardneg_in_batch_weight",
    "num_hard_negs",
    "learning_rate",
    "warmup_ratio",
    "weight_decay",
    "epochs",
    "effective_batch_size",
    "per_device_batch_size",
    "grad_accum",
    "world_size",
)
KD_KEYS = (
    "teacher_dir",
    "teacher_md5",
    "kd_dims",
    "kd_directions",
    "kd_include_hardnegs",
    "relation_weight",
    "margin_weight",
    "anchor_weight",
    "column_weight",
    "teacher_temperature",
    "anchor_dim",
    "calibration_dim",
    "student_temperatures",
)


def _weights(n: int, raw: list[float] | None) -> list[float]:
    if raw is None:
        return [1.0 / n] * n
    if len(raw) != n:
        raise ValueError(f"--weights length {len(raw)} != --adapters {n}")
    wsum = float(sum(raw))
    if wsum <= 0 or any(w < 0 for w in raw):
        raise ValueError("--weights must be non-negative and sum to > 0")
    return [float(w) / wsum for w in raw]


def _check_configs(adapters: list[str], configs: list[dict]) -> tuple[dict, list[int] | None]:
    reference = {k: configs[0][k] for k in CONTRACT if k in configs[0]}
    for adapter, config in zip(adapters[1:], configs[1:]):
        changed = sorted(k for k in reference if config.get(k) != reference[k])
        if changed:
            raise ValueError(
                f"recipe mismatch for {adapter}: "
                + ", ".join(f"{k}={reference[k]!r}->{config[k]!r}" for k in changed)
            )
    head_dims = configs[0].get("head_dims")
    for adapter, config in zip(adapters[1:], configs[1:]):
        if config.get("head_dims") != head_dims:
            raise ValueError(f"head_dims mismatch for {adapter}: {head_dims!r} -> {config.get('head_dims')!r}")
        if head_dims is not None:
            for key in KD_KEYS:
                if config.get(key) != configs[0].get(key):
                    raise ValueError(f"KD mismatch for {adapter}: {key}")
    if head_dims is not None:
        head_dims = [int(d) for d in head_dims]
    return reference, head_dims


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    args = ap.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    n = len(args.adapters)
    weights = _weights(n, args.weights)
    configs = []
    for adapter in args.adapters:
        path = Path(adapter) / "run_config.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        configs.append(json.loads(path.read_text(encoding="utf-8")))
    reference, head_dims = _check_configs(args.adapters, configs)
    col_dim = int(reference["col_dim"])

    def apply_shape(cfg):
        if head_dims is not None:
            cfg.head_dims = list(head_dims)
            cfg.dim = max(head_dims)
        else:
            cfg.dim = col_dim
        return cfg

    def load_base() -> ColQwen3_5:
        cfg = apply_shape(Qwen3_5Config.from_pretrained(args.base))
        return ColQwen3_5.from_pretrained(args.base, config=cfg, torch_dtype=dtype)

    acc = None
    shape = f"head_dims={head_dims}" if head_dims is not None else f"col_dim={col_dim}"
    for i, (adapter, w) in enumerate(zip(args.adapters, weights)):
        print(f"[{i + 1}/{n}] merge {adapter}  w={w:.6f}  ({shape})")
        if (Path(adapter) / "adapter_config.json").is_file():
            base = load_base()
            merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
            sd = merged.state_dict()
            del base, merged
        else:
            cfg = apply_shape(Qwen3_5Config.from_pretrained(adapter))
            full = ColQwen3_5.from_pretrained(adapter, config=cfg, torch_dtype=dtype)
            sd = full.state_dict()
            del full
        if acc is None:
            acc = {k: v.detach().to(torch.float32).mul(w) for k, v in sd.items()}
        else:
            for k, v in sd.items():
                acc[k].add_(v.detach().to(torch.float32), alpha=w)
        del sd
        torch.cuda.empty_cache()

    model = ColQwen3_5.from_pretrained(
        args.base,
        config=apply_shape(Qwen3_5Config.from_pretrained(args.base)),
        torch_dtype=torch.float32,
    )
    model.load_state_dict(acc, strict=True)

    output = forbid_venv_path(args.output, "output")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(args.output)
    ColQwen3_5Processor.from_pretrained(args.base).save_pretrained(args.output)
    merged_config = dict(configs[0])
    merged_config.update(
        output_dir=str(output.resolve()),
        run_name=output.name,
        merged_from=[str(Path(p).resolve()) for p in args.adapters],
        merge_weights=weights,
        merge_method="full_state_dict_mean" if args.weights is None else "full_state_dict_weighted",
        num_seeds=n,
        soup_arms=[
            {
                "adapter": str(Path(p).resolve()),
                "weight": w,
                "hardneg_subdir": c.get("hardneg_subdir"),
                "judged_pos": c.get("judged_pos"),
                "all_pos": c.get("all_pos"),
                "train_samples": c.get("train_samples"),
                "seed": c.get("seed"),
            }
            for p, c, w in zip(args.adapters, configs, weights)
        ],
    )
    (output / "run_config.json").write_text(
        json.dumps(merged_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("merged ->", args.output)


if __name__ == "__main__":
    main()
