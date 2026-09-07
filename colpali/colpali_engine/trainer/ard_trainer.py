"""ARD trainer: prefix-MRL relation distillation for MaxSim retrieval.

Gathers queries so each rank scores every global query against its local
documents, then reconstructs the full MaxSim square. That yields both the
query-to-candidate rows and the candidate-to-query columns of one matrix.

The frozen teacher is loaded at full bf16 weight, ``eval()`` + ``no_grad``,
and shares the student's batch (same tokenizer, patch 16 / merge 2). It scores
queries, positives, and (by default) hard negatives. Negatives from every rank
form one global relation pool.

``sample_weight`` and ``query_group_id`` come from the all-pos hard-negative rows.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from torch.distributed.nn.functional import all_gather  # PyTorch >= 2.1
from transformers import TrainerCallback

from colpali_engine.data.dataset import ColPaliEngineDataset
from colpali_engine.data.task_consistent_sampler import TaskConsistentDistributedSampler
from colpali_engine.models.qwen3_5.colqwen3_5.modeling_colqwen3_5 import set_active_head
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer
from colpali_engine.utils.maxsim import maxsim_inbatch, maxsim_kd

SCORE_TOL = 1e-3


def resolve_calibration_dim(calibration_dim: int, kd_dims: Sequence[int]) -> int:
    """Validate the explicit calibration probe width; never infer it from order."""
    dim = int(calibration_dim)
    normalized = tuple(sorted({int(d) for d in kd_dims}))
    if dim not in normalized:
        raise ValueError(f"calibration_dim={dim} is not in kd_dims={normalized}")
    return dim


class _PrefixRestoreCallback(TrainerCallback):
    """Undo AdamW decay/update on the frozen anchor rows during head warmup."""

    def __init__(self, snapshots: list[tuple[torch.nn.Parameter, torch.Tensor]], steps: int):
        self.snapshots = snapshots
        self.steps = int(steps)

    def on_optimizer_step(self, args, state, control, **kwargs):
        if int(state.global_step) < self.steps:
            with torch.no_grad():
                for parameter, snapshot in self.snapshots:
                    parameter[: snapshot.size(0)].copy_(
                        snapshot.to(device=parameter.device, dtype=parameter.dtype)
                    )
        return control


class _ARDStateCallback(TrainerCallback):
    """Persist non-parameter gradient-calibration state in every checkpoint."""

    def __init__(self, trainer: "ARDTrainer"):
        self.trainer = trainer

    def on_save(self, args, state, control, **kwargs):
        if int(args.process_index) != 0:
            return control
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        payload = {
            "calibrated_relation_scale": self.trainer._calibrated_relation_scale,
            "last_calibration_step": self.trainer._last_calibration_step,
        }
        (checkpoint / "ard_state.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        return control


class ARDTrainer(ContrastiveTrainer):
    """Args:
        teacher_model: frozen single-head ``ColQwen3_5``, or ``None`` for a
            retrieval-only run. Never wrapped in DDP: it has no gradients.
        head_dims: student head widths.
        kd_include_hardnegs: add every rank's explicit negatives to the global
            row candidate pool. Costs a third teacher forward.
        log_head_terms: emit per-head retrieval/KD scalars to the tracker.
    """

    def __init__(
        self,
        *args,
        teacher_model=None,
        head_dims: Sequence[int] = (),
        anchor_dim: int = 128,
        use_anchor_teacher: bool = True,
        kd_include_hardnegs: bool = True,
        task_consistent_batches: bool = True,
        calibration_dim: int = 128,
        gradient_target_ratio: float = 0.5,
        gradient_calibration_steps: int = 100,
        gradient_calibration_interval: int = 10,
        gradient_scale_min: float = 0.05,
        gradient_scale_max: float = 20.0,
        gradient_scale_ema: float = 0.9,
        gradient_diagnostics: bool = True,
        gradient_diagnostic_steps: int = 100,
        gradient_diagnostic_interval: int = 10,
        head_warmup_steps: int = 100,
        log_head_terms: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.head_dims = tuple(sorted({int(d) for d in head_dims}))
        if not self.head_dims:
            raise ValueError("ARDTrainer requires head_dims")
        if int(anchor_dim) not in self.head_dims:
            raise ValueError(f"anchor_dim={anchor_dim} is not in {self.head_dims}")
        self.teacher_model = teacher_model
        self.anchor_dim = int(anchor_dim)
        self.use_anchor_teacher = bool(use_anchor_teacher)
        self.kd_include_hardnegs = bool(kd_include_hardnegs)
        self.task_consistent_batches = bool(task_consistent_batches)
        self.calibration_dim = resolve_calibration_dim(
            calibration_dim, self.loss_func.kd_dims
        )
        self.gradient_target_ratio = float(gradient_target_ratio)
        self.gradient_calibration_steps = int(gradient_calibration_steps)
        self.gradient_calibration_interval = int(gradient_calibration_interval)
        self.gradient_scale_min = float(gradient_scale_min)
        self.gradient_scale_max = float(gradient_scale_max)
        self.gradient_scale_ema = float(gradient_scale_ema)
        self.gradient_diagnostics = bool(gradient_diagnostics)
        self.gradient_diagnostic_steps = int(gradient_diagnostic_steps)
        self.gradient_diagnostic_interval = int(gradient_diagnostic_interval)
        self.head_warmup_steps = int(head_warmup_steps)
        if self.gradient_target_ratio < 0:
            raise ValueError("gradient_target_ratio must be non-negative")
        if not 0 <= self.gradient_scale_ema < 1:
            raise ValueError("gradient_scale_ema must be in [0, 1)")
        if self.gradient_diagnostic_steps < 0:
            raise ValueError("gradient_diagnostic_steps must be non-negative")
        if self.gradient_diagnostic_interval <= 0:
            raise ValueError("gradient_diagnostic_interval must be positive")
        self.log_head_terms = bool(log_head_terms)
        self._pending_logs: Dict[str, float] = {}
        self._calibrated_relation_scale = 1.0
        self._last_calibration_step = -1
        anchor_snapshots = [
            (parameter, parameter[: self.anchor_dim].detach().clone())
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
            and "custom_text_proj" in name
            and parameter.dim() in (1, 2)
            and parameter.size(0) >= self.anchor_dim
        ]
        if self.head_warmup_steps > 0:
            if not anchor_snapshots:
                raise RuntimeError("head warmup could not find the trainable prefix projection")
            self.add_callback(
                _PrefixRestoreCallback(anchor_snapshots, self.head_warmup_steps)
            )
        self.add_callback(_ARDStateCallback(self))
        if self.teacher_model is not None:
            self.teacher_model.to(self.args.device)
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad_(False)

    def train(self, resume_from_checkpoint=None, *args, **kwargs):
        if isinstance(resume_from_checkpoint, (str, Path)):
            state_path = Path(resume_from_checkpoint) / "ard_state.json"
            if state_path.is_file():
                payload = json.loads(state_path.read_text(encoding="utf-8"))
                self._calibrated_relation_scale = float(
                    payload["calibrated_relation_scale"]
                )
                self._last_calibration_step = int(payload["last_calibration_step"])
                self.loss_func.calibrated_relation_scale = (
                    self._calibrated_relation_scale
                )
                print(
                    f"[ard] restored kd_scale={self._calibrated_relation_scale:.6f} "
                    f"from {state_path}",
                    flush=True,
                )
            elif self.gradient_calibration_steps > 0:
                step_text = Path(resume_from_checkpoint).name.rsplit("-", 1)[-1]
                if step_text.isdigit() and int(step_text) > 0:
                    raise FileNotFoundError(
                        f"resume checkpoint lacks ARD calibration state: {state_path}"
                    )
        return super().train(
            *args,
            resume_from_checkpoint=resume_from_checkpoint,
            **kwargs,
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if int(self.state.global_step) < self.head_warmup_steps:
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                if "lora_" in name:
                    parameter.grad = None
                elif "custom_text_proj" in name:
                    if parameter.grad.dim() >= 1 and parameter.grad.size(0) >= self.anchor_dim:
                        parameter.grad[: self.anchor_dim].zero_()
            if int(self.state.global_step) == 0:
                self._pending_logs["ard/head_warmup"] = 1.0
        return loss

    def _get_train_sampler(self, dataset=None):
        target = dataset if dataset is not None else self.train_dataset
        task_ids = getattr(target, "task_ids", None)
        if self.task_consistent_batches:
            if task_ids is None:
                raise RuntimeError(
                    "task-consistent batching requested but train_dataset has no task_ids"
                )
            return TaskConsistentDistributedSampler(
                task_ids=task_ids,
                local_batch_size=self.args.per_device_train_batch_size,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                seed=self.args.seed,
                drop_last=self.args.dataloader_drop_last,
                # accelerator.prepare shards consecutive DataLoader batches.
                # Yield the shared global order here to avoid double sharding.
                shard_by_rank=False,
            )
        return super()._get_train_sampler(dataset)

    # ------------------------------------------------------------ collectives

    @property
    def _world(self) -> int:
        return self.accelerator.num_processes

    def _gather_rows(
        self, t: torch.Tensor, differentiable: bool, pad_dim: Optional[int] = None
    ) -> torch.Tensor:
        """Concatenate one tensor per rank along dim 0, in rank order.

        Rank order matters: it is what makes ``offset = rank * Bl`` address this
        rank's own rows in the gathered result.
        """
        if self._world == 1:
            return t
        if pad_dim is not None:
            # Each rank pads queries to its own batch maximum, so widths differ.
            t = self.accelerator.pad_across_processes(t, dim=pad_dim, pad_index=0, pad_first=True)
        t = t.contiguous()
        if differentiable:
            return torch.cat(all_gather(t), dim=0)
        buf = [torch.empty_like(t) for _ in range(self._world)]
        torch.distributed.all_gather(buf, t)
        return torch.cat(buf, dim=0)

    def _gather_cols(self, block: torch.Tensor, differentiable: bool) -> torch.Tensor:
        """Concatenate one score block per rank along dim 1 -> the global square.

        Every rank holds ``[Bg, Bl]`` with the same ``Bl`` because the dataloader
        runs with ``drop_last=True``, so nothing needs padding here.
        """
        if self._world == 1:
            return block
        block = block.contiguous()
        if differentiable:
            return torch.cat(all_gather(block), dim=1)
        buf = [torch.empty_like(block) for _ in range(self._world)]
        torch.distributed.all_gather(buf, block)
        return torch.cat(buf, dim=1)

    def _gather_id_rows(self, values: torch.Tensor, pad_value: int = -1) -> torch.Tensor:
        if self._world == 1:
            return values
        if values.dim() > 1:
            values = self.accelerator.pad_across_processes(
                values,
                dim=1,
                pad_index=pad_value,
                pad_first=False,
            )
        buf = [torch.empty_like(values) for _ in range(self._world)]
        torch.distributed.all_gather(buf, values.contiguous())
        return torch.cat(buf, dim=0)

    # ---------------------------------------------------------------- scoring

    @staticmethod
    def _normalize(scores: torch.Tensor, query_lengths: torch.Tensor, tag: str) -> torch.Tensor:
        """Divide MaxSim sums by each query's token count, pinning scores to [-1, 1].

        ``query_lengths`` comes from ``attention_mask``. After per-token L2
        normalization a real token can carry a zero in component 0; counting
        nonzeros there would under-count, especially on narrow prefixes.

        Queries index dim 0 in every matrix here (``[Bg, Bl]``, ``[Bl, N]``), so the
        divisor always broadcasts along dim 1.
        """
        out = scores / query_lengths.unsqueeze(1).to(scores.dtype)
        low, high = torch.aminmax(out.detach())
        if low < -1 - SCORE_TOL or high > 1 + SCORE_TOL:
            print(
                f"[kd] {tag}: normalized scores outside [-1, 1]: "
                f"min={low.item():.4f} max={high.item():.4f} (tol={SCORE_TOL})",
                flush=True,
            )
        return out

    def _score_tower(
        self,
        q_all: torch.Tensor,
        doc_local: torch.Tensor,
        neg_local: Optional[torch.Tensor],
        q_local: torch.Tensor,
        qlen_all: torch.Tensor,
        qlen_local: torch.Tensor,
        offset: int,
        differentiable: bool,
    ) -> Dict[str, torch.Tensor]:
        """One head (or the teacher's single head) -> every matrix the loss needs."""
        batch = doc_local.size(0)
        block = self._normalize(maxsim_inbatch(q_all, doc_local), qlen_all, "in-batch")  # [Bg, Bl]
        square = self._gather_cols(block, differentiable)  # [Bg, Bg]
        rows = square[offset : offset + batch]  # [Bl, Bg]

        idx = torch.arange(batch, device=block.device)
        pos = block[idx + offset, idx]  # [Bl]: the diagonal of this rank's own slice

        out = {"pos": pos, "rows": rows, "cols": block, "kd_rows": rows}
        if neg_local is not None:
            neg = self._normalize(maxsim_kd(q_local, neg_local), qlen_local, "hardneg")  # [Bl, N]
            out["neg"] = neg
            if self.kd_include_hardnegs:
                flat_neg = neg_local.flatten(0, 1)
                neg_block = self._normalize(
                    maxsim_inbatch(q_all, flat_neg),
                    qlen_all,
                    "global-hardneg",
                )
                neg_square = self._gather_cols(neg_block, differentiable)
                out["kd_rows"] = torch.cat(
                    [rows, neg_square[offset : offset + batch]],
                    dim=1,
                )
        return out

    # ------------------------------------------------------------ input split

    def _split_inputs(self, inputs):
        query_inputs = {
            k[len(self.query_prefix) :]: v for k, v in inputs.items() if k.startswith(self.query_prefix)
        }
        doc_inputs = {
            k[len(self.pos_prefix) :]: v for k, v in inputs.items() if k.startswith(self.pos_prefix)
        }
        neg_inputs = None
        num_negs = 0
        if "neg_doc_input_ids" in inputs:
            num_negs = inputs["neg_doc_input_ids"].size(1)
            neg_inputs = self._reshape_neg_doc_inputs(inputs)
        return query_inputs, doc_inputs, neg_inputs, num_negs

    def _candidate_masks(
        self,
        group_ids: Optional[torch.Tensor],
        positive_ids: Optional[torch.Tensor],
        positive_sets: Optional[torch.Tensor],
        negative_ids: Optional[torch.Tensor],
        slot_mask: Optional[torch.Tensor],
        offset: int,
        batch: int,
        num_negs: int,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Mask known positives, duplicate targets and invalid negative slots.

        All-pos shards give every (query, positive) pair its own row, so two
        rows in one batch can share a query with different positives. Unmasked,
        row i would be pushed away from row j's page even though that page is a
        real positive for the same query. The mask covers both KL directions,
        and the teacher sees the same mask so the two distributions stay
        comparable.
        """
        global_mask = None
        if group_ids is not None:
            gid_all = self._gather_rows(group_ids, differentiable=False)
            global_mask = gid_all.unsqueeze(1) == gid_all.unsqueeze(0)
            global_mask.fill_diagonal_(False)

        pos_all = None
        pos_sets_all = None
        if positive_ids is not None and positive_sets is not None:
            pos_all = self._gather_id_rows(positive_ids)
            pos_sets_all = self._gather_id_rows(positive_sets)
            known_positive = (
                pos_sets_all.unsqueeze(2) == pos_all.view(1, 1, -1)
            ).any(dim=1)
            known_positive.fill_diagonal_(False)
            global_mask = (
                known_positive
                if global_mask is None
                else global_mask.logical_or(known_positive)
            )

        row_mask = None if global_mask is None else global_mask[offset : offset + batch].clone()
        col_mask = None if global_mask is None else global_mask[:, offset : offset + batch].clone()
        kd_row_mask = row_mask
        if self.kd_include_hardnegs and num_negs > 0:
            if negative_ids is not None:
                neg_all = self._gather_id_rows(negative_ids).flatten()
            else:
                neg_all = torch.full(
                    (self._world * batch * num_negs,),
                    -2,
                    dtype=torch.long,
                    device=(
                        positive_ids.device
                        if positive_ids is not None
                        else slot_mask.device
                        if slot_mask is not None
                        else self.args.device
                    ),
                )
            invalid = torch.zeros_like(neg_all, dtype=torch.bool)
            if slot_mask is not None:
                invalid = self._gather_rows(slot_mask <= 0, differentiable=False).flatten()
            neg_mask = invalid.unsqueeze(0).expand(batch, -1).clone()
            local_positive_sets = positive_sets
            if local_positive_sets is not None:
                matches = (
                    local_positive_sets.unsqueeze(2) == neg_all.view(1, 1, -1)
                ) & (local_positive_sets.unsqueeze(2) >= 0)
                neg_mask.logical_or_(matches.any(dim=1))
            if row_mask is None:
                row_mask = torch.zeros(
                    batch,
                    self._world * batch,
                    dtype=torch.bool,
                    device=neg_mask.device,
                )
                col_mask = torch.zeros(
                    self._world * batch,
                    batch,
                    dtype=torch.bool,
                    device=neg_mask.device,
                )
            kd_row_mask = torch.cat([row_mask, neg_mask], dim=1)
        return row_mask, col_mask, kd_row_mask

    @staticmethod
    def _peft_adapter_model(model):
        """Reach the PEFT wrapper without unwrapping through it to ColQwen3_5."""
        current = model
        for _ in range(8):
            if callable(getattr(current, "disable_adapter", None)):
                return current
            inner = getattr(current, "module", None)
            if inner is None or inner is current:
                inner = getattr(current, "_orig_mod", None)
            if inner is None or inner is current:
                break
            current = inner
        raise TypeError(f"no PEFT adapter wrapper under {type(model).__name__}")

    def _calibrate_relation_scale(
        self,
        parts: Dict[str, torch.Tensor],
        probe_tensors: Sequence[torch.Tensor],
    ) -> None:
        step = int(self.state.global_step)
        if (
            self.gradient_target_ratio <= 0
            or step >= self.gradient_calibration_steps
            or step == self._last_calibration_step
            or step % max(1, self.gradient_calibration_interval) != 0
        ):
            return
        kd_objective = (
            self.loss_func.relation_weight * parts["relation"]
            + self.loss_func.margin_weight * parts["margin"]
            + self.loss_func.anchor_weight * parts["anchor"]
        )
        if not kd_objective.requires_grad:
            return
        task_grads = torch.autograd.grad(
            parts["task"],
            probe_tensors,
            retain_graph=True,
            allow_unused=True,
        )
        kd_grads = torch.autograd.grad(
            kd_objective,
            probe_tensors,
            retain_graph=True,
            allow_unused=True,
        )
        task_used = [grad for grad in task_grads if grad is not None]
        kd_used = [grad for grad in kd_grads if grad is not None]
        if not task_used or not kd_used:
            return
        # Probe gradients stop at shared d128 query/document representations, so
        # they do not invoke DDP parameter hooks (DDP does not support
        # parameter-level torch.autograd.grad). RMS-reduce local squared norms to
        # keep one deterministic scale on all ranks.
        task_sq = sum(grad.float().square().sum() for grad in task_used)
        kd_sq = sum(grad.float().square().sum() for grad in kd_used)
        if self._world > 1:
            torch.distributed.all_reduce(task_sq)
            torch.distributed.all_reduce(kd_sq)
            task_sq.div_(self._world)
            kd_sq.div_(self._world)
        task_norm = task_sq.sqrt()
        kd_norm = kd_sq.sqrt()
        target = self.gradient_target_ratio * float(task_norm) / max(float(kd_norm), 1e-12)
        target = min(max(target, self.gradient_scale_min), self.gradient_scale_max)
        if self._last_calibration_step < 0:
            self._calibrated_relation_scale = target
        else:
            self._calibrated_relation_scale = (
                self.gradient_scale_ema * self._calibrated_relation_scale
                + (1.0 - self.gradient_scale_ema) * target
            )
        self._last_calibration_step = step
        self.loss_func.calibrated_relation_scale = self._calibrated_relation_scale
        self._pending_logs.update(
            {
                "ard/grad_task_repr": float(task_norm),
                "ard/grad_kd_repr": float(kd_norm),
                "ard/kd_scale": self._calibrated_relation_scale,
            }
        )

    def _diagnose_dim_gradients(
        self,
        dim_parts: Dict[int, Dict[str, torch.Tensor]],
        q_student: Dict[int, torch.Tensor],
        d_student: Dict[int, torch.Tensor],
    ) -> None:
        """Log per-width objective representation gradients without changing loss.

        This deliberately probes output representations rather than DDP
        parameters. Parameter-level ``autograd.grad`` can fire DDP hooks outside
        the normal backward. Diagnostics run only in the configured early-step
        window and retain the graph for the real loss backward.
        """
        step = int(self.state.global_step)
        if (
            not self.gradient_diagnostics
            or step >= self.gradient_diagnostic_steps
            or step % self.gradient_diagnostic_interval != 0
        ):
            return

        def objective(dim: int) -> torch.Tensor:
            terms = dim_parts[dim]
            value = self.loss_func.head_weights[dim] * terms["task"]
            if dim in self.loss_func.kd_dims:
                kd_weight = self.loss_func.kd_head_weights[dim]
                value = value + self._calibrated_relation_scale * kd_weight * (
                    self.loss_func.relation_weight * terms["relation"]
                    + self.loss_func.margin_weight * terms["margin"]
                )
            if "anchor" in terms:
                value = value + self._calibrated_relation_scale * (
                    self.loss_func.anchor_weight * terms["anchor"]
                )
            return value

        def repr_grads(dim: int) -> tuple[torch.Tensor, torch.Tensor]:
            grads = torch.autograd.grad(
                objective(dim),
                (q_student[dim], d_student[dim]),
                retain_graph=True,
                allow_unused=True,
            )
            return tuple(
                torch.zeros_like(tensor) if grad is None else grad
                for grad, tensor in zip(grads, (q_student[dim], d_student[dim]))
            )

        reference = self.calibration_dim
        ref_grads = repr_grads(reference)
        for dim in self.head_dims:
            grads = ref_grads if dim == reference else repr_grads(dim)
            shared = min(dim, reference)
            grad_shared = torch.cat(
                [grad[..., :shared].float().reshape(-1) for grad in grads]
            )
            ref_shared = torch.cat(
                [grad[..., :shared].float().reshape(-1) for grad in ref_grads]
            )
            stats = torch.stack(
                (
                    sum(grad.float().square().sum() for grad in grads),
                    grad_shared.square().sum(),
                    ref_shared.square().sum(),
                    (grad_shared * ref_shared).sum(),
                )
            )
            if self._world > 1:
                torch.distributed.all_reduce(stats)
                stats.div_(self._world)
            norm = stats[0].clamp_min(0).sqrt()
            cosine = stats[3] / (stats[1] * stats[2]).clamp_min(1e-24).sqrt()
            self._pending_logs.update(
                {
                    f"ard/grad_total_repr/d{dim}": float(norm),
                    f"ard/grad_cos_d{dim}_vs_d{reference}": float(cosine),
                }
            )

    # ------------------------------------------------------------ compute_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        query_inputs, doc_inputs, neg_inputs, num_negs = self._split_inputs(inputs)

        slot_mask = inputs.get(ColPaliEngineDataset.MASK_KEY)
        judged = getattr(self.train_dataset, "judged_pos", False)
        all_pos = getattr(self.train_dataset, "all_pos", False)
        if (judged or all_pos) and slot_mask is None:
            raise RuntimeError(
                "judged-pos / all_pos 已启用，但 batch 没有 hardneg_slot_mask；"
                "拒绝回退到对全部 hardneg 槽求均值"
            )
        sample_weight = inputs.get(ColPaliEngineDataset.WEIGHT_KEY)
        group_ids = inputs.get(ColPaliEngineDataset.GROUP_KEY)
        positive_ids = inputs.get(ColPaliEngineDataset.POS_ID_KEY)
        positive_sets = inputs.get(ColPaliEngineDataset.POS_IDS_KEY)
        negative_ids = inputs.get(ColPaliEngineDataset.NEG_IDS_KEY)
        if all_pos and (sample_weight is None or group_ids is None):
            raise RuntimeError(
                "all_pos 已启用，但 batch 缺 sample_weight / query_group_id；"
                "拒绝在不加权、不屏蔽同组的情况下训练"
            )

        qlen_local = query_inputs["attention_mask"].sum(dim=1)
        batch = qlen_local.size(0)
        offset = self.accelerator.process_index * batch if self._world > 1 else 0

        # ---- student: one backbone and one maximum projection; heads are prefixes
        set_active_head(model, None)
        q_student = model(**query_inputs)
        d_student = model(**doc_inputs)
        n_student = model(**neg_inputs) if neg_inputs is not None else None

        qlen_all = self._gather_rows(qlen_local, differentiable=False)
        row_mask, col_mask, kd_row_mask = self._candidate_masks(
            group_ids,
            positive_ids,
            positive_sets,
            negative_ids,
            slot_mask,
            offset,
            batch,
            num_negs,
        )

        student: Dict[int, Dict[str, torch.Tensor]] = {}
        for d in self.head_dims:
            q_d = q_student[d]
            neg_d = None
            if n_student is not None:
                neg_d = self._reshape_neg_doc_outputs(n_student[d], num_negs)
            student[d] = self._score_tower(
                q_all=self._gather_rows(q_d, differentiable=True, pad_dim=1),
                doc_local=d_student[d],
                neg_local=neg_d,
                q_local=q_d,
                qlen_all=qlen_all,
                qlen_local=qlen_local,
                offset=offset,
                differentiable=True,
            )

        # ---- teacher: frozen, full width, same batch
        teacher = None
        if self.teacher_model is not None:
            with torch.no_grad():
                q_teacher = self.teacher_model(**query_inputs)
                d_teacher = self.teacher_model(**doc_inputs)
                neg_teacher = None
                if neg_inputs is not None and self.kd_include_hardnegs:
                    neg_teacher = self._reshape_neg_doc_outputs(
                        self.teacher_model(**neg_inputs), num_negs
                    )
                teacher = self._score_tower(
                    q_all=self._gather_rows(q_teacher, differentiable=False, pad_dim=1),
                    doc_local=d_teacher,
                    neg_local=neg_teacher,
                    q_local=q_teacher,
                    qlen_all=qlen_all,
                    qlen_local=qlen_local,
                    offset=offset,
                    differentiable=False,
                )

        # ---- frozen Preview anchor without duplicating model weights.
        # PEFT's disable_adapter context selects the immutable original module
        # behind modules_to_save and disables backbone LoRA at the same time.
        anchor_teacher = None
        if self.use_anchor_teacher and self.loss_func.anchor_weight > 0:
            peft_model = self._peft_adapter_model(model)
            with torch.no_grad(), peft_model.disable_adapter():
                set_active_head(model, self.anchor_dim)
                q_anchor = model(**query_inputs)
                d_anchor = model(**doc_inputs)
                anchor_teacher = self._score_tower(
                    q_all=self._gather_rows(q_anchor, differentiable=False, pad_dim=1),
                    doc_local=d_anchor,
                    neg_local=None,
                    q_local=q_anchor,
                    qlen_all=qlen_all,
                    qlen_local=qlen_local,
                    offset=offset,
                    differentiable=False,
                )
            set_active_head(model, None)

        pos_idx = torch.arange(batch, device=qlen_local.device) + offset
        parts, logs = self.loss_func.compute_parts(
            student=student,
            teacher=teacher,
            anchor_teacher=anchor_teacher,
            anchor_dim=self.anchor_dim,
            pos_idx=pos_idx,
            slot_mask=slot_mask,
            row_mask=row_mask,
            kd_row_mask=kd_row_mask,
            col_mask=col_mask,
            sample_weight=sample_weight,
        )
        self._calibrate_relation_scale(
            parts,
            (
                q_student[self.calibration_dim],
                d_student[self.calibration_dim],
            ),
        )
        self._diagnose_dim_gradients(
            self.loss_func.last_dim_parts,
            q_student,
            d_student,
        )
        self.loss_func.last_dim_parts = {}
        loss = self.loss_func.combine(parts, self._calibrated_relation_scale)
        logs.update(
            {
                "ard/task": float(parts["task"].detach()),
                "ard/relation": float(parts["relation"].detach()),
                "ard/margin": float(parts["margin"].detach()),
                "ard/anchor": float(parts["anchor"].detach()),
                "ard/kd_scale": self._calibrated_relation_scale,
            }
        )
        if self.log_head_terms:
            self._pending_logs.update(logs)
        return (loss, (q_student, d_student)) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if self._pending_logs:
            logs.update(self._pending_logs)
            self._pending_logs = {}
        return super().log(logs, *args, **kwargs)
