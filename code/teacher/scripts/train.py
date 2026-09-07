#!/usr/bin/env python3
"""Minimal local ColQwen3.5 LoRA trainer for explicit train-only sources."""

from __future__ import annotations

import argparse
import json
import os
import time
import sys
from pathlib import Path

_SHARED = Path(__file__).resolve().parents[2] / "shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

import torch
from peft import LoraConfig
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import TrainingArguments, set_seed

from data_loader import (
    ALLOWED_SOURCES,
    _default_dataset_cache_dir,
    build_hardneg_dataset,
    build_train_dataset,
)
from paths import forbid_venv_path
from colpali_engine.loss.late_interaction_losses import ColbertLoss, ColbertNegativeCELoss
from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor
from transformers.models.qwen3_5 import Qwen3_5Config
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig

TARGET_MODULES = (
    r"(.*(model)(?!.*visual).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj|"
    r"in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj).*$)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sources", nargs="*", default=[], choices=ALLOWED_SOURCES,
                        help="Optional ablation filter on the source column; empty = full corpus.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-samples-per-source", type=int, default=0)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=4.57e-5)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--max-visual-tokens", type=int, default=1024)
    parser.add_argument("--col-dim", type=int, default=512,
                        help="custom_text_proj output dim (4096 for EVIE-8B).")
    parser.add_argument("--dataloader-workers", type=int, default=2)
    parser.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        default=4,
        help="Batches prefetched by each DataLoader worker; ignored when workers=0.",
    )
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.197)
    parser.add_argument("--loss-temperature", type=float, default=0.02)
    parser.add_argument("--hardneg-in-batch-weight", type=float, default=0.5)
    parser.add_argument("--resume-from-checkpoint", default="",
                        help="Checkpoint path, or 'latest' to resume the newest output checkpoint.")
    parser.add_argument("--attn", choices=("flash_attention_2", "sdpa", "eager"), default="flash_attention_2")
    parser.add_argument("--grad-checkpointing", choices=("on", "off"), default="off")
    parser.add_argument(
        "--bidirectional-attention",
        choices=("on", "off"),
        default="on",
        help="on = encoder-ize full-attention layers (ColEmbed V2).",
    )
    parser.add_argument(
        "--hardneg-root",
        default="",
        help="Hardneg output with queries/. corpus/ is optional; without it images load from --data-root.",
    )
    parser.add_argument("--num-hard-negs", type=int, default=2)
    parser.add_argument(
        "--use-hardnegatives",
        choices=("on", "off"),
        default="on",
        help="When --hardneg-root is set: on=ColbertNegativeCELoss; off=same rows, in-batch only.",
    )
    parser.add_argument(
        "--report-to",
        default="none",
        help="Comma-separated metric trackers, e.g. wandb,tensorboard.",
    )
    parser.add_argument("--logging-dir", default=None, help="TensorBoard event output directory.")
    parser.add_argument("--run-name", default=None, help="Run name for the tracker (e.g. wandb).")
    return parser.parse_args()


def resolve_pretrained(spec: str) -> str:
    path = Path(spec)
    return str(path.resolve()) if path.exists() else spec


def prepare_output(path: Path, resume_requested: bool) -> None:
    path = forbid_venv_path(path, "output-dir")
    prepared_by_launcher = os.environ.get("EVIE_OUTPUT_PREPARED") == "1"
    if path.exists() and any(path.iterdir()) and not resume_requested and not prepared_by_launcher:
        raise FileExistsError(f"Refusing to overwrite a non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    base_model = resolve_pretrained(args.base_model)
    data_root = Path(args.data_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root is missing: {data_root}")
    if args.hardneg_root:
        hn = Path(args.hardneg_root).resolve()
        subdir = os.environ.get("HARDNEG_SUBDIR", "judged")
        if not (hn / subdir).is_dir():
            raise FileNotFoundError(f"hardneg-root needs {subdir}/: {hn}")
    prepare_output(output_dir, resume_requested=bool(args.resume_from_checkpoint))
    os.environ.setdefault("HF_DATASETS_CACHE", str(_default_dataset_cache_dir()))
    set_seed(args.seed)

    print("== building train-only query/image pairs ==")
    use_hardneg = bool(args.hardneg_root) and args.use_hardnegatives == "on"
    if args.hardneg_root:
        train_dataset = build_hardneg_dataset(
            hardneg_root=args.hardneg_root,
            data_root=data_root,
            num_negatives=args.num_hard_negs,
            max_samples=args.max_samples_per_source,
            use_negatives=use_hardneg,
        )
    else:
        train_dataset = build_train_dataset(
            data_root=data_root,
            sources=args.sources,
            max_samples_per_source=args.max_samples_per_source,
        )

    print("== loading processor and bf16 base model ==")
    processor = ColQwen3_5Processor.from_pretrained(
        base_model,
        max_num_visual_tokens=args.max_visual_tokens,
    )
    # Raw Qwen3.5 has no ColBERT head. Set config.dim before build so
    # custom_text_proj is created at --col-dim; missing keys stay randomly
    # initialized. Forward L2-normalizes each token, so init scale washes out.
    model_config = Qwen3_5Config.from_pretrained(base_model)
    model_config.dim = args.col_dim
    model = ColQwen3_5.from_pretrained(
        base_model,
        config=model_config,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn,
    )
    print(f"[model] custom_text_proj = Linear(-> {args.col_dim}), fresh full-rank head")
    try:
        model.rope_deltas = None
    except AttributeError:
        pass

    if args.bidirectional_attention == "on":
        model.enable_bidirectional_attention()
        print("[model] bidirectional attention enabled on full-attention layers")

    use_gc = args.grad_checkpointing == "on"
    if use_gc:
        model.enable_input_require_grads()

    report_to = [item.strip() for item in args.report_to.split(",") if item.strip()]
    if not report_to or report_to == ["none"]:
        report_to = []
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=use_gc,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_gc else None,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.dataloader_workers > 0,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor if args.dataloader_workers > 0 else None,
        dataloader_drop_last=True,
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        seed=args.seed,
        data_seed=args.seed,
        ddp_find_unused_parameters=False,
        report_to=report_to,
        logging_dir=args.logging_dir,
        run_name=args.run_name,
    )
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules=TARGET_MODULES,
        modules_to_save=["custom_text_proj"],  # fresh head: full-rank, not LoRA
    )
    if use_hardneg:
        judged = bool(getattr(train_dataset, "judged_pos", False))
        all_pos = bool(getattr(train_dataset, "all_pos", False))
        mode = ", judged-pos" if judged else (", all-pos" if all_pos else "")
        print(
            f"== Loss: ColbertNegativeCELoss (hard_negs={args.num_hard_negs}{mode}) =="
        )
        loss_func = ColbertNegativeCELoss(
            temperature=args.loss_temperature,
            normalize_scores=True,
            use_smooth_max=False,
            pos_aware_negative_filtering=True,
            in_batch_term_weight=args.hardneg_in_batch_weight,
        )
    else:
        print("== Loss: ColbertLoss (in-batch only) ==")
        loss_func = ColbertLoss(
            temperature=args.loss_temperature,
            normalize_scores=True,
            use_smooth_max=False,
        )
    trainer = ColModelTraining(
        ColModelTrainingConfig(
            output_dir=str(output_dir),
            processor=processor,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=None,
            run_eval=False,
            loss_func=loss_func,
            tr_args=training_args,
            peft_config=lora,
        )
    )
    print("== training ==")
    started = time.time()
    resume = args.resume_from_checkpoint or None
    if resume == "latest":
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.rsplit("-", 1)[-1]),
        )
        if not checkpoints:
            raise FileNotFoundError(f"no checkpoint-* found under {output_dir}")
        resume = str(checkpoints[-1])
    training_args.resume_from_checkpoint = resume
    trainer.train()
    trainer.save()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    try:
        n_samples = len(train_dataset)
    except TypeError:
        n_samples = None
    config = vars(args) | {
        "base_model": str(base_model),
        "data_root": str(data_root),
        "framework": "minimal-colqwen35-lora",
        "hardneg_subdir": (
            os.environ.get("HARDNEG_SUBDIR", "judged") if args.hardneg_root else None
        ),
        "judged_pos": bool(getattr(train_dataset, "judged_pos", False)),
        "all_pos": bool(getattr(train_dataset, "all_pos", False)),
        "world_size": world_size,
        "effective_batch_size": args.per_device_batch_size * args.grad_accum * world_size,
        "train_samples": n_samples,
        "train_runtime_seconds": round(time.time() - started, 1),
        "total_params": sum(p.numel() for p in trainer.model.parameters()),
        "trainable_params": sum(
            p.numel() for p in trainer.model.parameters() if p.requires_grad
        ),
    }
    if int(os.environ.get("RANK", "0")) == 0:
        (output_dir / "run_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    print(f"== complete: {output_dir} ==")


@record
def run_with_error_recording() -> None:
    """Persist the original failing DDP rank's traceback for torchrun."""
    try:
        main()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    run_with_error_recording()
