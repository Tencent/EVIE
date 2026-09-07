"""EVIE-ARD loss for prefix-MRL late-interaction retrieval.

The task objective supervises every configured prefix. Relation distillation is
restricted to selected capacity-compatible prefixes and aligns both rows and
columns of the same asymmetric MaxSim matrix. Explicit hard negatives extend the
row candidate pool and additionally receive a teacher-margin regression loss.

``ARDTrainer`` calibrates
the explicit relation scale from task/KD gradient norms during warmup.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

KD_DIRECTIONS = ("both", "row", "column", "none")
MASK_LOGIT = -1e4


def _normalized_weights(
    dims: Sequence[int],
    supplied: Optional[Mapping[int, float]],
) -> dict[int, float]:
    raw = {int(d): float((supplied or {}).get(int(d), 1.0)) for d in dims}
    if any(v < 0 for v in raw.values()):
        raise ValueError(f"head weights must be non-negative, got {raw}")
    total = sum(raw.values())
    if total <= 0:
        raise ValueError(f"head weights must contain a positive value, got {raw}")
    return {d: value / total for d, value in raw.items()}


class ARDLoss(nn.Module):
    """Anchor-preserving, capacity-aware relation distillation."""

    def __init__(
        self,
        head_dims: Sequence[int],
        kd_dims: Sequence[int],
        temperature: float = 0.02,
        teacher_temperature: float = 0.13,
        student_temperatures: Optional[Mapping[int, float]] = None,
        relation_weight: float = 1.0,
        margin_weight: float = 0.25,
        anchor_weight: float = 0.25,
        column_weight: float = 1.0,
        in_batch_term_weight: float = 0.5,
        kd_directions: str = "both",
        confidence_floor: float = 0.1,
        teacher_wrong_factor: float = 0.25,
        head_weights: Optional[Mapping[int, float]] = None,
        kd_head_weights: Optional[Mapping[int, float]] = None,
        pos_aware_negative_filtering: bool = True,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        super().__init__()
        self.head_dims = tuple(sorted({int(d) for d in head_dims}))
        self.kd_dims = tuple(sorted({int(d) for d in kd_dims}))
        if not self.head_dims:
            raise ValueError("head_dims must not be empty")
        if not self.kd_dims or not set(self.kd_dims).issubset(self.head_dims):
            raise ValueError(
                f"kd_dims must be a non-empty subset of head_dims: {self.kd_dims} vs {self.head_dims}"
            )
        if temperature <= 0 or teacher_temperature <= 0:
            raise ValueError("temperatures must be positive")
        if kd_directions not in KD_DIRECTIONS:
            raise ValueError(f"kd_directions must be one of {KD_DIRECTIONS}")
        if kd_directions in ("both", "column") and column_weight <= 0:
            raise ValueError("column_weight must be positive when column KD is enabled")
        if not 0 <= in_batch_term_weight <= 1:
            raise ValueError("in_batch_term_weight must be in [0, 1]")
        if not 0 <= confidence_floor <= 1 or not 0 <= teacher_wrong_factor <= 1:
            raise ValueError("confidence controls must be in [0, 1]")
        for name, value in (
            ("relation_weight", relation_weight),
            ("margin_weight", margin_weight),
            ("anchor_weight", anchor_weight),
            ("column_weight", column_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")

        self.temperature = float(temperature)
        self.teacher_temperature = float(teacher_temperature)
        supplied_temps = student_temperatures or {}
        self.student_temperatures = {
            d: float(supplied_temps.get(d, teacher_temperature)) for d in self.kd_dims
        }
        if any(value <= 0 for value in self.student_temperatures.values()):
            raise ValueError(f"student temperatures must be positive: {self.student_temperatures}")

        self.relation_weight = float(relation_weight)
        self.margin_weight = float(margin_weight)
        self.anchor_weight = float(anchor_weight)
        self.column_weight = float(column_weight)
        self.in_batch_term_weight = float(in_batch_term_weight)
        self.kd_directions = kd_directions
        self.want_row = kd_directions in ("both", "row")
        self.want_column = kd_directions in ("both", "column")
        self.confidence_floor = float(confidence_floor)
        self.teacher_wrong_factor = float(teacher_wrong_factor)
        self.head_weights = _normalized_weights(self.head_dims, head_weights)
        self.kd_head_weights = _normalized_weights(self.kd_dims, kd_head_weights)
        self.pos_aware_negative_filtering = bool(pos_aware_negative_filtering)
        self.filter_threshold = float(filter_threshold)
        self.filter_factor = float(filter_factor)
        self.calibrated_relation_scale = 1.0
        # Ephemeral graph tensors consumed by ARDTrainer diagnostics in the same
        # forward. They are not parameters, state, or part of the loss value.
        self.last_dim_parts: dict[int, dict[str, torch.Tensor]] = {}

    @staticmethod
    def _reduce(
        per_sample: torch.Tensor,
        sample_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if sample_weight is None:
            return per_sample.mean()
        weights = sample_weight.to(dtype=per_sample.dtype, device=per_sample.device)
        if weights.shape != per_sample.shape:
            raise ValueError(
                f"sample_weight {tuple(weights.shape)} does not match {tuple(per_sample.shape)}"
            )
        return (weights * per_sample).sum() / weights.sum().clamp(min=1e-6)

    @staticmethod
    def _mask_logits(scores: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return scores
        if mask.shape != scores.shape:
            raise ValueError(f"mask {tuple(mask.shape)} does not match scores {tuple(scores.shape)}")
        return scores.masked_fill(mask, MASK_LOGIT)

    def _teacher_confidence(
        self,
        teacher: torch.Tensor,
        pos_idx: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        logits = self._mask_logits(teacher.float() / self.teacher_temperature, mask)
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1)
        if mask is None:
            valid = torch.full(
                (teacher.size(0),),
                teacher.size(1),
                dtype=torch.float32,
                device=teacher.device,
            )
        else:
            valid = (~mask).sum(dim=1).float()
        clarity = 1.0 - entropy / valid.clamp(min=2).log()
        clarity = clarity.clamp(0.0, 1.0)
        confidence = self.confidence_floor + (1.0 - self.confidence_floor) * clarity
        correct = logits.argmax(dim=1) == pos_idx
        confidence = torch.where(correct, confidence, confidence * self.teacher_wrong_factor)
        return confidence.detach()

    def _kl_rows(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        pos_idx: torch.Tensor,
        mask: Optional[torch.Tensor],
        student_temperature: float,
        use_confidence: bool,
    ) -> torch.Tensor:
        student_logits = self._mask_logits(student.float() / student_temperature, mask)
        teacher_logits = self._mask_logits(
            teacher.float() / self.teacher_temperature,
            mask,
        )
        log_q = F.log_softmax(student_logits, dim=1)
        log_p = F.log_softmax(teacher_logits, dim=1)
        per_sample = (log_p.exp() * (log_p - log_q)).sum(dim=1)
        if use_confidence:
            per_sample = per_sample * self._teacher_confidence(teacher, pos_idx, mask)
        return per_sample

    def _filter_high_negatives(
        self,
        scores: torch.Tensor,
        pos_idx: torch.Tensor,
    ) -> torch.Tensor:
        idx = torch.arange(scores.size(0), device=scores.device)
        pos = scores[idx, pos_idx]
        hot = scores > self.filter_threshold * pos.unsqueeze(1)
        hot[idx, pos_idx] = False
        return torch.where(hot, scores * self.filter_factor, scores)

    def _task_loss(
        self,
        head: Dict[str, torch.Tensor],
        pos_idx: torch.Tensor,
        slot_mask: Optional[torch.Tensor],
        row_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pos = head["pos"].float()
        neg = head.get("neg")
        if neg is None:
            hard = torch.zeros_like(pos)
        else:
            per_slot = F.softplus((neg.float() - pos.unsqueeze(1)) / self.temperature)
            if slot_mask is None:
                hard = per_slot.mean(dim=1)
            else:
                mask = slot_mask.to(dtype=per_slot.dtype, device=per_slot.device)
                if mask.shape != per_slot.shape:
                    raise ValueError(
                        f"slot_mask {tuple(mask.shape)} does not match neg {tuple(per_slot.shape)}"
                    )
                hard = (per_slot * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        if self.in_batch_term_weight == 0:
            return hard
        rows = self._mask_logits(head["rows"].float(), row_mask)
        if self.pos_aware_negative_filtering:
            rows = self._filter_high_negatives(rows, pos_idx)
        ce = F.cross_entropy(rows / self.temperature, pos_idx, reduction="none")
        return (
            hard * (1.0 - self.in_batch_term_weight)
            + ce * self.in_batch_term_weight
        )

    def _relation_loss(
        self,
        head: Dict[str, torch.Tensor],
        teacher: Dict[str, torch.Tensor],
        pos_idx: torch.Tensor,
        kd_row_mask: Optional[torch.Tensor],
        col_mask: Optional[torch.Tensor],
        student_temperature: float,
        use_confidence: bool = True,
        want_row: Optional[bool] = None,
        want_column: Optional[bool] = None,
    ) -> torch.Tensor:
        use_row = self.want_row if want_row is None else bool(want_row)
        use_column = self.want_column if want_column is None else bool(want_column)
        terms = []
        if use_row:
            terms.append(
                self._kl_rows(
                    head["kd_rows"],
                    teacher["kd_rows"],
                    pos_idx,
                    kd_row_mask,
                    student_temperature,
                    use_confidence,
                )
            )
        if use_column:
            terms.append(
                self.column_weight
                * self._kl_rows(
                    head["cols"].transpose(0, 1),
                    teacher["cols"].transpose(0, 1),
                    pos_idx,
                    None if col_mask is None else col_mask.transpose(0, 1),
                    student_temperature,
                    use_confidence,
                )
            )
        if not terms:
            return torch.zeros_like(head["pos"], dtype=torch.float32)
        # Keep the KD scale invariant when one direction is disabled.
        return sum(terms) / sum(
            (1.0 if use_row else 0.0, self.column_weight if use_column else 0.0)
        )

    def _margin_loss(
        self,
        head: Dict[str, torch.Tensor],
        teacher: Dict[str, torch.Tensor],
        slot_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        student_neg = head.get("neg")
        teacher_neg = teacher.get("neg")
        if student_neg is None or teacher_neg is None:
            return torch.zeros_like(head["pos"], dtype=torch.float32)
        student_margin = head["pos"].float().unsqueeze(1) - student_neg.float()
        teacher_margin = teacher["pos"].float().unsqueeze(1) - teacher_neg.float()
        confidence = torch.sigmoid(teacher_margin / self.teacher_temperature).detach()
        per_slot = F.smooth_l1_loss(
            student_margin,
            teacher_margin,
            reduction="none",
        ) * confidence
        if slot_mask is None:
            return per_slot.mean(dim=1)
        mask = slot_mask.to(dtype=per_slot.dtype, device=per_slot.device)
        return (per_slot * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def compute_parts(
        self,
        student: Dict[int, Dict[str, torch.Tensor]],
        teacher: Optional[Dict[str, torch.Tensor]],
        anchor_teacher: Optional[Dict[str, torch.Tensor]],
        anchor_dim: int,
        pos_idx: torch.Tensor,
        slot_mask: Optional[torch.Tensor] = None,
        row_mask: Optional[torch.Tensor] = None,
        kd_row_mask: Optional[torch.Tensor] = None,
        col_mask: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> tuple[dict[str, torch.Tensor], Dict[str, float]]:
        missing = [d for d in self.head_dims if d not in student]
        if missing:
            raise ValueError(f"student is missing heads {missing}")

        task_total = None
        relation_total = None
        margin_total = None
        logs: Dict[str, float] = {}
        self.last_dim_parts = {}
        for d in self.head_dims:
            task_per_sample = self._task_loss(student[d], pos_idx, slot_mask, row_mask)
            task = self._reduce(task_per_sample, sample_weight)
            weighted_task = task * self.head_weights[d]
            task_total = weighted_task if task_total is None else task_total + weighted_task
            logs[f"task/d{d}"] = float(task.detach())

            relation = task * 0.0
            margin = task * 0.0
            if teacher is not None and d in self.kd_dims:
                if self.kd_directions == "none":
                    relation = task * 0.0
                else:
                    relation = self._reduce(
                        self._relation_loss(
                            student[d],
                            teacher,
                            pos_idx,
                            kd_row_mask,
                            col_mask,
                            self.student_temperatures[d],
                        ),
                        sample_weight,
                    )
                margin = self._reduce(
                    self._margin_loss(student[d], teacher, slot_mask),
                    sample_weight,
                )
                kd_weight = self.kd_head_weights[d]
                relation_total = (
                    relation * kd_weight
                    if relation_total is None
                    else relation_total + relation * kd_weight
                )
                margin_total = (
                    margin * kd_weight
                    if margin_total is None
                    else margin_total + margin * kd_weight
                )
                logs[f"rel/d{d}"] = float(relation.detach())
                logs[f"margin/d{d}"] = float(margin.detach())
            self.last_dim_parts[d] = {
                "task": task,
                "relation": relation,
                "margin": margin,
            }

        zero = next(iter(student.values()))["pos"].float().sum() * 0.0
        task_total = zero if task_total is None else task_total
        relation_total = zero if relation_total is None else relation_total
        margin_total = zero if margin_total is None else margin_total

        anchor_total = zero
        if anchor_teacher is not None and self.anchor_weight > 0:
            if anchor_dim not in student:
                raise ValueError(f"anchor_dim={anchor_dim} is not a trained prefix")
            anchor_head = dict(student[anchor_dim])
            # The capability-teacher row pool may include global hard negatives;
            # the adapter-disabled Preview anchor intentionally preserves only
            # the original in-batch d128 relation geometry.
            anchor_head["kd_rows"] = anchor_head["rows"]
            anchor_total = self._reduce(
                self._relation_loss(
                    anchor_head,
                    anchor_teacher,
                    pos_idx,
                    row_mask,
                    col_mask,
                    self.student_temperatures.get(
                        anchor_dim,
                        self.teacher_temperature,
                    ),
                    use_confidence=False,
                    want_row=True,
                    want_column=True,
                ),
                sample_weight,
            )
            logs[f"anchor/d{anchor_dim}"] = float(anchor_total.detach())
            self.last_dim_parts[anchor_dim]["anchor"] = anchor_total

        parts = {
            "task": task_total,
            "relation": relation_total,
            "margin": margin_total,
            "anchor": anchor_total,
        }
        return parts, logs

    def combine(
        self,
        parts: Mapping[str, torch.Tensor],
        calibrated_relation_scale: float = 1.0,
    ) -> torch.Tensor:
        return (
            parts["task"]
            + float(calibrated_relation_scale)
            * (
                self.relation_weight * parts["relation"]
                + self.margin_weight * parts["margin"]
                + self.anchor_weight * parts["anchor"]
            )
        )

