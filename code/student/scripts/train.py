#!/usr/bin/env python3
"""Prefix-MRL student trainer with ARD.

One maximum projection is initialized from Preview d128. A frozen EVIE-8B
teacher transfers row/column MaxSim relations and hard-negative margins; the
adapter-disabled Preview path anchors d128.
"""

from __future__ import annotations

import argparse
import hashlib
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
from colpali_engine.loss.ard import ARDLoss
from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor
from transformers.models.qwen3_5 import Qwen3_5Config
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.trainer.ard_trainer import ARDTrainer

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
    parser.add_argument("--learning-rate", type=float, default=1.5e-5,
                        help="Warm-start default. Teacher from-scratch uses 4.57e-5.")
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--max-visual-tokens", type=int, default=1024)
    parser.add_argument("--col-dim", type=int, default=512,
                        help="custom_text_proj output dim (unused when --head-dims is set).")
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
    parser.add_argument(
        "--head-dims",
        default="",
        help="Comma-separated Prefix-MRL widths, e.g. '64,128,256,512,1024,2048'. "
        "Empty turns off prefixes and the teacher.",
    )
    parser.add_argument(
        "--anchor-dim",
        type=int,
        default=128,
        help="Preview projection width copied into the Prefix-MRL prefix rows.",
    )
    parser.add_argument(
        "--teacher-dir",
        default="",
        help="Directory of a frozen full-weight teacher (EVIE-8B). "
        "Empty together with --head-dims is retrieval-only.",
    )
    parser.add_argument(
        "--teacher-md5",
        default="",
        help="Expected md5 of the teacher model.safetensors; verified before training starts.",
    )
    parser.add_argument(
        "--kd-dims",
        default="64,128,256,512,1024,2048",
        help="Prefixes that receive relation distillation.",
    )
    parser.add_argument("--calibration-dim", type=int, default=128)
    parser.add_argument("--teacher-temperature", type=float, default=0.13)
    parser.add_argument(
        "--student-temperatures",
        default="64:0.13,128:0.13,256:0.13,512:0.13,1024:0.13,2048:0.13",
    )
    parser.add_argument("--relation-weight", type=float, default=1.0)
    parser.add_argument("--margin-weight", type=float, default=0.25)
    parser.add_argument("--anchor-weight", type=float, default=0.25)
    parser.add_argument("--column-weight", type=float, default=1.0)
    parser.add_argument("--confidence-floor", type=float, default=0.1)
    parser.add_argument("--teacher-wrong-factor", type=float, default=0.25)
    parser.add_argument("--head-weights", default="")
    parser.add_argument("--kd-head-weights", default="")
    parser.add_argument(
        "--kd-directions",
        choices=("both", "row", "column", "none"),
        default="both",
    )
    parser.add_argument("--kd-include-hardnegs", choices=("on", "off"), default="on")
    parser.add_argument("--anchor-teacher", choices=("on", "off"), default="on")
    parser.add_argument("--task-consistent-batches", choices=("on", "off"), default="on")
    parser.add_argument("--gradient-target-ratio", type=float, default=0.5)
    parser.add_argument("--gradient-calibration-steps", type=int, default=100)
    parser.add_argument("--gradient-calibration-interval", type=int, default=10)
    parser.add_argument("--gradient-scale-min", type=float, default=0.05)
    parser.add_argument("--gradient-scale-max", type=float, default=20.0)
    parser.add_argument("--gradient-scale-ema", type=float, default=0.9)
    parser.add_argument("--gradient-diagnostics", choices=("on", "off"), default="on")
    parser.add_argument("--gradient-diagnostic-steps", type=int, default=100)
    parser.add_argument("--gradient-diagnostic-interval", type=int, default=10)
    parser.add_argument("--head-warmup-steps", type=int, default=100)
    return parser.parse_args()


def resolve_pretrained(spec: str) -> str:
    path = Path(spec)
    return str(path.resolve()) if path.exists() else spec


def parse_head_dims(spec: str) -> tuple[int, ...]:
    if not spec.strip():
        return ()
    dims = tuple(sorted({int(x) for x in spec.replace(" ", "").split(",") if x}))
    if not dims or dims[0] <= 0:
        raise ValueError(f"--head-dims must be positive integers, got {spec!r}")
    return dims


def parse_dim_map(spec: str, allowed: tuple[int, ...], name: str) -> dict[int, float]:
    if not spec.strip():
        return {}
    values: dict[int, float] = {}
    for item in spec.replace(" ", "").split(","):
        if not item:
            continue
        try:
            raw_dim, raw_value = item.split(":", 1)
            dim, value = int(raw_dim), float(raw_value)
        except ValueError as exc:
            raise ValueError(f"{name} expects d:value entries, got {item!r}") from exc
        if dim not in allowed:
            raise ValueError(f"{name} dim {dim} is not in {allowed}")
        values[dim] = value
    return values


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_teacher(teacher_dir: Path, expected_md5: str, attn: str, bidirectional: str):
    """Load the frozen teacher at full bf16 weight, eval + no_grad.

    Full weights only (merged soup, not base + adapter). Optional md5 check
    rejects a wrong teacher before training starts.

    Returns ``(teacher, verified_md5)``. The digest is reused for
    ``run_config.json`` so every rank does not hash the same file twice.
    """
    weights = teacher_dir / "model.safetensors"
    shards = sorted(teacher_dir.glob("model-*-of-*.safetensors"))
    if not weights.is_file() and not shards:
        raise FileNotFoundError(f"teacher has no model*.safetensors: {teacher_dir}")
    verified_md5 = None
    if expected_md5:
        if not weights.is_file():
            raise ValueError(
                f"--teacher-md5 given but teacher is sharded ({len(shards)} shards); "
                "cannot verify a single-file md5"
            )
        verified_md5 = file_md5(weights)
        if verified_md5 != expected_md5:
            raise ValueError(
                f"teacher md5 mismatch: expected {expected_md5}, got {verified_md5} ({weights})"
            )
        print(f"[teacher] md5 verified: {verified_md5}")

    config = Qwen3_5Config.from_pretrained(teacher_dir)
    if getattr(config, "head_dims", None):
        raise ValueError("teacher must be single-head; found head_dims in its config")
    teacher = ColQwen3_5.from_pretrained(
        teacher_dir,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn,
    )
    try:
        teacher.rope_deltas = None
    except AttributeError:
        pass
    if bidirectional == "on":
        teacher.enable_bidirectional_attention()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(
        f"[teacher] {teacher_dir.name}: dim={teacher.dim} "
        f"bidir={bidirectional} params={sum(p.numel() for p in teacher.parameters()):,}"
    )
    return teacher, verified_md5


def assert_shared_batch(student, teacher, processor) -> None:
    checks = {
        "patch_size": (student.patch_size, teacher.patch_size, processor.image_processor.patch_size),
        "spatial_merge_size": (
            student.spatial_merge_size,
            teacher.spatial_merge_size,
            processor.image_processor.merge_size,
        ),
    }
    for name, (s, t, p) in checks.items():
        if not (int(s) == int(t) == int(p)):
            raise ValueError(
                f"{name} disagrees: student={s} teacher={t} processor={p}"
            )
    for name in ("image_token_id", "video_token_id", "vision_start_token_id", "vocab_size"):
        s = getattr(student.config, name, None)
        t = getattr(teacher.config, name, None)
        if s != t:
            raise ValueError(f"{name} disagrees: student={s} teacher={t}")
    print(
        f"[teacher] shared-batch OK (patch={student.patch_size} "
        f"merge={student.spatial_merge_size} image_token_id={student.config.image_token_id})"
    )


def prepare_output(path: Path, resume_requested: bool) -> None:
    path = forbid_venv_path(path, "output-dir")
    prepared_by_launcher = os.environ.get("EVIE_OUTPUT_PREPARED") == "1"
    if path.exists() and any(path.iterdir()) and not resume_requested and not prepared_by_launcher:
        raise FileExistsError(f"Refusing to overwrite a non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    head_dims = parse_head_dims(args.head_dims)
    kd_dims = parse_head_dims(args.kd_dims)
    if head_dims:
        if args.anchor_dim not in head_dims:
            raise ValueError(f"--anchor-dim {args.anchor_dim} must be in --head-dims {head_dims}")
        if not kd_dims or not set(kd_dims).issubset(head_dims):
            raise ValueError(f"--kd-dims {kd_dims} must be a non-empty subset of {head_dims}")
        if args.calibration_dim not in kd_dims:
            raise ValueError(
                f"--calibration-dim {args.calibration_dim} must be in --kd-dims {kd_dims}"
            )
        student_temperatures = parse_dim_map(
            args.student_temperatures, kd_dims, "--student-temperatures"
        )
        head_weights = parse_dim_map(args.head_weights, head_dims, "--head-weights")
        kd_head_weights = parse_dim_map(args.kd_head_weights, kd_dims, "--kd-head-weights")
    else:
        student_temperatures = {}
        head_weights = {}
        kd_head_weights = {}
    teacher_dir = Path(args.teacher_dir).resolve() if args.teacher_dir else None
    if teacher_dir is not None:
        if not head_dims:
            raise ValueError("--teacher-dir requires --head-dims")
        if not teacher_dir.is_dir():
            raise FileNotFoundError(f"teacher dir is missing: {teacher_dir}")
        if args.margin_weight > 0 and args.kd_include_hardnegs != "on":
            raise ValueError("--margin-weight > 0 requires --kd-include-hardnegs on")
    output_dir = Path(args.output_dir).resolve()
    base_model = resolve_pretrained(args.base_model)
    data_root = Path(args.data_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root is missing: {data_root}")
    if args.hardneg_root:
        hn = Path(args.hardneg_root).resolve()
        subdir = os.environ.get("HARDNEG_SUBDIR", "allpos")
        if not (hn / subdir).is_dir():
            raise FileNotFoundError(f"hardneg-root needs {subdir}/: {hn}")
    prepare_output(output_dir, resume_requested=bool(args.resume_from_checkpoint))
    os.environ.setdefault("HF_DATASETS_CACHE", str(_default_dataset_cache_dir()))
    set_seed(args.seed)

    print("== building train-only query/image pairs ==")
    use_hardneg = bool(args.hardneg_root) and args.use_hardnegatives == "on"
    if teacher_dir is not None and args.margin_weight > 0 and not use_hardneg:
        raise ValueError("--margin-weight > 0 requires --hardneg-root and --use-hardnegatives on")
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
    # Load Preview in its original d128 shape first, then expand to Prefix-MRL.
    # Setting head_dims before from_pretrained would shape-mismatch and drop
    # the deployed 128-d projection.
    model_config = Qwen3_5Config.from_pretrained(base_model)
    if not head_dims:
        model_config.dim = args.col_dim
    model = ColQwen3_5.from_pretrained(
        base_model,
        config=model_config,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn,
    )
    if head_dims:
        model.enable_prefix_mrl(head_dims, anchor_dim=args.anchor_dim)
        print(
            f"[model] prefix MRL = Linear({model.hidden_size_for_heads}->{max(head_dims)}), "
            f"dims={list(head_dims)}, copied Preview rows [0:{args.anchor_dim})"
        )
    else:
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
    head_modules = model.head_module_names()
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules=TARGET_MODULES,
        modules_to_save=head_modules,
    )
    print(f"[lora] modules_to_save={head_modules}")

    teacher = None
    teacher_md5 = None
    trainer_cls = None
    trainer_kwargs: dict = {}
    if head_dims:
        if teacher_dir is not None:
            print(f"== loading frozen teacher: {teacher_dir} ==")
            teacher, teacher_md5 = load_teacher(
                teacher_dir, args.teacher_md5, args.attn, args.bidirectional_attention
            )
            assert_shared_batch(model, teacher, processor)
        else:
            print("== no teacher: Prefix-MRL retrieval-only ==")
        print(
            f"== Loss: ARD (prefixes={list(head_dims)}, kd_dims={list(kd_dims)}, "
            f"tau_T={args.teacher_temperature}, tau_S={student_temperatures}, "
            f"relation={args.relation_weight}, margin={args.margin_weight}, "
            f"anchor={args.anchor_weight}, dirs={args.kd_directions}, "
            f"global_hardnegs={args.kd_include_hardnegs}, "
            f"task_consistent={args.task_consistent_batches}) =="
        )
        loss_func = ARDLoss(
            head_dims=head_dims,
            kd_dims=kd_dims,
            temperature=args.loss_temperature,
            teacher_temperature=args.teacher_temperature,
            student_temperatures=student_temperatures,
            relation_weight=args.relation_weight if teacher is not None else 0.0,
            margin_weight=args.margin_weight if teacher is not None else 0.0,
            anchor_weight=args.anchor_weight if args.anchor_teacher == "on" else 0.0,
            column_weight=args.column_weight,
            in_batch_term_weight=args.hardneg_in_batch_weight,
            kd_directions=args.kd_directions if teacher is not None else "none",
            confidence_floor=args.confidence_floor,
            teacher_wrong_factor=args.teacher_wrong_factor,
            head_weights=head_weights,
            kd_head_weights=kd_head_weights,
            pos_aware_negative_filtering=True,
        )
        trainer_cls = ARDTrainer
        trainer_kwargs = {
            "teacher_model": teacher,
            "head_dims": head_dims,
            "anchor_dim": args.anchor_dim,
            "use_anchor_teacher": args.anchor_teacher == "on",
            "kd_include_hardnegs": args.kd_include_hardnegs == "on",
            "task_consistent_batches": args.task_consistent_batches == "on",
            "gradient_target_ratio": args.gradient_target_ratio,
            "gradient_calibration_steps": args.gradient_calibration_steps,
            "gradient_calibration_interval": args.gradient_calibration_interval,
            "gradient_scale_min": args.gradient_scale_min,
            "gradient_scale_max": args.gradient_scale_max,
            "gradient_scale_ema": args.gradient_scale_ema,
            "calibration_dim": args.calibration_dim,
            "gradient_diagnostics": args.gradient_diagnostics == "on",
            "gradient_diagnostic_steps": args.gradient_diagnostic_steps,
            "gradient_diagnostic_interval": args.gradient_diagnostic_interval,
            "head_warmup_steps": args.head_warmup_steps,
        }
    elif use_hardneg:
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
            trainer_cls=trainer_cls,
            trainer_kwargs=trainer_kwargs,
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
        "framework": "evie-ard-prefix-mrl",
        "hardneg_subdir": (
            os.environ.get("HARDNEG_SUBDIR", "allpos") if args.hardneg_root else None
        ),
        "judged_pos": bool(getattr(train_dataset, "judged_pos", False)),
        "all_pos": bool(getattr(train_dataset, "all_pos", False)),
        "head_dims": list(head_dims) if head_dims else None,
        "mrl_prefix": bool(head_dims),
        "anchor_dim": args.anchor_dim if head_dims else None,
        "kd_dims": list(kd_dims) if head_dims else None,
        "calibration_dim": args.calibration_dim if head_dims else None,
        "student_temperatures_parsed": (
            loss_func.student_temperatures if head_dims else None
        ),
        "col_dim": max(head_dims) if head_dims else args.col_dim,
        "teacher_dir": str(teacher_dir) if teacher_dir else None,
        "teacher_md5": teacher_md5,
        "teacher_col_dim": int(teacher.dim) if teacher is not None else None,
        "kd_active": teacher is not None,
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
