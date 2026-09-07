import torch
import torch.nn.functional as F  # noqa: N812
from torch.nn import CrossEntropyLoss

from colpali_engine.utils.maxsim import maxsim_inbatch, maxsim_kd


class ColbertModule(torch.nn.Module):
    """
    Base module for ColBERT losses, handling shared utilities and hyperparameters.

    Args:
        max_batch_size (int): Maximum batch size for pre-allocating index buffer.
        tau (float): Temperature for smooth-max approximation.
        norm_tol (float): Tolerance for score normalization bounds.
        filter_threshold (float): Ratio threshold for pos-aware negative filtering.
        filter_factor (float): Multiplicative factor to down-weight high negatives.
    """

    def __init__(
        self,
        max_batch_size: int = 1024,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        super().__init__()
        self.register_buffer("idx_buffer", torch.arange(max_batch_size), persistent=False)
        self.tau = tau
        self.norm_tol = norm_tol
        self.filter_threshold = filter_threshold
        self.filter_factor = filter_factor

    def _get_idx(self, batch_size: int, offset: int, device: torch.device):
        """
        Retrieve index and positive index tensors for in-batch losses.
        """
        idx = self.idx_buffer[:batch_size].to(device)
        return idx, idx + offset

    def _smooth_max(self, scores: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Compute smooth max via log-sum-exp along a given dimension.
        """
        return self.tau * torch.logsumexp(scores / self.tau, dim=dim)

    def _apply_normalization(self, scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Normalize scores by query lengths and enforce bounds.

        Args:
            scores (Tensor): Unnormalized score matrix [B, C].
            lengths (Tensor): Query lengths [B].

        Returns:
            Tensor: Normalized scores.

        Raises:
            ValueError: If normalized scores exceed tolerance.
        """
        if scores.ndim == 2:
            normalized = scores / lengths.unsqueeze(1)
        else:
            normalized = scores / lengths

        return normalized

    def _aggregate(
        self,
        scores_raw: torch.Tensor,
        use_smooth_max: bool,
        dim_max: int,
        dim_sum: int,
    ) -> torch.Tensor:
        """
        Aggregate token-level scores into document-level.

        Args:
            scores_raw (Tensor): Raw scores tensor.
            use_smooth_max (bool): Use smooth-max if True.
            dim_max (int): Dimension to perform max/logsumexp.
            dim_sum (int): Dimension to sum over after max.
        """
        if use_smooth_max:
            return self._smooth_max(scores_raw, dim=dim_max).sum(dim=dim_sum)
        return scores_raw.amax(dim=dim_max).sum(dim=dim_sum)

    def _inbatch_scores(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute the in-batch MaxSim score matrix and apply optional length normalization.

        Routes through the fused late-interaction kernel when ``use_smooth_max`` is False;
        smooth-max keeps the logsumexp path since the kernel only exposes hard max.
        """
        if self.use_smooth_max:
            raw = torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings)
            scores = self._aggregate(raw, True, dim_max=3, dim_sum=2)
        else:
            scores = maxsim_inbatch(query_embeddings, doc_embeddings)
        if self.normalize_scores:
            lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1)
            scores = self._apply_normalization(scores, lengths)
        return scores

    def _filter_high_negatives(self, scores: torch.Tensor, pos_idx: torch.Tensor) -> None:
        """
        Down-weight negatives whose score exceeds a fraction of the positive score.

        Args:
            scores (Tensor): In-batch score matrix [B, B].
            pos_idx (Tensor): Positive indices for each query in batch.
        """
        batch_size = scores.size(0)
        idx = self.idx_buffer[:batch_size].to(scores.device)
        pos_scores = scores[idx, pos_idx]
        thresh = self.filter_threshold * pos_scores.unsqueeze(1)
        mask = scores > thresh
        mask[idx, pos_idx] = False
        scores[mask] *= self.filter_factor


class ColbertLoss(ColbertModule):
    """
    InfoNCE loss for late interaction (ColBERT) without explicit negatives.

    Args:
        temperature (float): Scaling factor for logits.
        normalize_scores (bool): Normalize scores by query lengths.
        use_smooth_max (bool): Use log-sum-exp instead of amax.
        pos_aware_negative_filtering (bool): Apply pos-aware negative filtering.
    """

    def __init__(
        self,
        temperature: float = 0.02,
        normalize_scores: bool = True,
        use_smooth_max: bool = False,
        pos_aware_negative_filtering: bool = False,
        max_batch_size: int = 1024,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        super().__init__(max_batch_size, tau, norm_tol, filter_threshold, filter_factor)
        self.temperature = temperature
        self.normalize_scores = normalize_scores
        self.use_smooth_max = use_smooth_max
        self.pos_aware_negative_filtering = pos_aware_negative_filtering
        self.ce_loss = CrossEntropyLoss()

    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        offset: int = 0,
        reduction: str = "mean",
        group_ids_local: torch.Tensor = None,
        group_ids_global: torch.Tensor = None,
    ) -> torch.Tensor:
        """InfoNCE. reduction='none' returns per-row loss [B] for all-pos sample weighting."""
        scores = self._inbatch_scores(query_embeddings, doc_embeddings)

        batch_size = scores.size(0)
        idx, pos_idx = self._get_idx(batch_size, offset, scores.device)

        if self.pos_aware_negative_filtering:
            self._filter_high_negatives(scores, pos_idx)

        if group_ids_local is not None and group_ids_global is not None:
            same = group_ids_local.view(-1, 1) == group_ids_global.view(1, -1).to(group_ids_local.device)
            same[idx, pos_idx] = False
            scores = scores.masked_fill(same, float("-inf"))

        logits = scores / self.temperature
        if reduction == "none":
            return F.cross_entropy(logits, pos_idx, reduction="none")
        return self.ce_loss(logits, pos_idx)


class ColbertNegativeCELoss(ColbertModule):
    """
    InfoNCE loss with explicit negative documents.

    Args:
        temperature (float): Scaling for logits.
        normalize_scores (bool): Normalize scores by query lengths.
        use_smooth_max (bool): Use log-sum-exp instead of amax.
        pos_aware_negative_filtering (bool): Apply pos-aware negative filtering.
        in_batch_term_weight (float): Add in-batch CE term (between 0 and 1).
    """

    def __init__(
        self,
        temperature: float = 0.02,
        normalize_scores: bool = True,
        use_smooth_max: bool = False,
        pos_aware_negative_filtering: bool = False,
        in_batch_term_weight: float = 0.5,
        max_batch_size: int = 1024,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        super().__init__(max_batch_size, tau, norm_tol, filter_threshold, filter_factor)
        self.temperature = temperature
        self.normalize_scores = normalize_scores
        self.use_smooth_max = use_smooth_max
        self.pos_aware_negative_filtering = pos_aware_negative_filtering
        self.in_batch_term_weight = in_batch_term_weight
        self.ce_loss = CrossEntropyLoss()
        self.inner_loss = ColbertLoss(
            temperature=temperature,
            normalize_scores=normalize_scores,
            use_smooth_max=use_smooth_max,
            pos_aware_negative_filtering=pos_aware_negative_filtering,
            max_batch_size=max_batch_size,
            tau=tau,
            norm_tol=norm_tol,
            filter_threshold=filter_threshold,
            filter_factor=filter_factor,
        )

    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        neg_doc_embeddings: torch.Tensor,
        offset: int = 0,
        slot_mask: torch.Tensor = None,
        sample_weight: torch.Tensor = None,
        group_ids_local: torch.Tensor = None,
        group_ids_global: torch.Tensor = None,
    ) -> torch.Tensor:
        """Hard-neg InfoNCE; optional per-row sample_weight (all-pos) and same-query mask."""
        lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1)
        pos_raw = torch.einsum(
            "bnd,bsd->bns", query_embeddings, doc_embeddings[offset : offset + neg_doc_embeddings.size(0)]
        )
        pos_scores = self._aggregate(pos_raw, self.use_smooth_max, dim_max=2, dim_sum=1)
        if self.use_smooth_max:
            # Smooth-max keeps the logsumexp path since the kernel only exposes hard max.
            neg_raw = torch.einsum("bnd,blsd->blns", query_embeddings, neg_doc_embeddings)
            neg_scores = self._aggregate(neg_raw, True, dim_max=3, dim_sum=2)
        else:
            neg_scores = maxsim_kd(query_embeddings, neg_doc_embeddings)

        if self.normalize_scores:
            pos_scores = self._apply_normalization(pos_scores, lengths)
            neg_scores = self._apply_normalization(neg_scores, lengths)

        per_slot = F.softplus((neg_scores - pos_scores.unsqueeze(1)) / self.temperature)

        # No sample_weight: scalar mean (judged-pos batches).
        if sample_weight is None:
            if slot_mask is None:
                loss = per_slot.mean()
            else:
                mask = slot_mask.to(dtype=per_slot.dtype, device=per_slot.device)[:, : per_slot.size(1)]
                loss = (per_slot * mask).sum() / mask.sum().clamp(min=1.0)
            if self.in_batch_term_weight > 0:
                loss_ib = self.inner_loss(query_embeddings, doc_embeddings, offset)
                loss = loss * (1 - self.in_batch_term_weight) + loss_ib * self.in_batch_term_weight
            return loss

        if slot_mask is None:
            hn = per_slot.mean(dim=1)
        else:
            mask = slot_mask.to(dtype=per_slot.dtype, device=per_slot.device)[:, : per_slot.size(1)]
            hn = (per_slot * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        if self.in_batch_term_weight > 0:
            ib = self.inner_loss(
                query_embeddings, doc_embeddings, offset,
                reduction="none",
                group_ids_local=group_ids_local,
                group_ids_global=group_ids_global,
            )  # [B]
            per_sample = hn * (1 - self.in_batch_term_weight) + ib * self.in_batch_term_weight
        else:
            per_sample = hn

        w = sample_weight.to(dtype=per_sample.dtype, device=per_sample.device)
        return (w * per_sample).sum() / w.sum().clamp(min=1e-6)


class ColbertListwiseKLLoss(ColbertModule):
    """Listwise KL over one positive and an ordered candidate list."""

    def __init__(
        self,
        temperature: float = 0.02,
        teacher_temperature: float = 1.0,
        normalize_scores: bool = True,
        use_smooth_max: bool = False,
        in_batch_term_weight: float = 0.5,
        max_batch_size: int = 1024,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        super().__init__(max_batch_size, tau, norm_tol, filter_threshold, filter_factor)
        if teacher_temperature <= 0:
            raise ValueError("teacher_temperature must be positive")
        if not 0 <= in_batch_term_weight <= 1:
            raise ValueError("in_batch_term_weight must be in [0, 1]")
        self.temperature = temperature
        self.teacher_temperature = teacher_temperature
        self.normalize_scores = normalize_scores
        self.use_smooth_max = use_smooth_max
        self.in_batch_term_weight = in_batch_term_weight
        self.inner_loss = ColbertLoss(
            temperature=temperature,
            normalize_scores=normalize_scores,
            use_smooth_max=use_smooth_max,
            max_batch_size=max_batch_size,
            tau=tau,
            norm_tol=norm_tol,
            filter_threshold=filter_threshold,
            filter_factor=filter_factor,
        )

    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        neg_doc_embeddings: torch.Tensor,
        listwise_grades: torch.Tensor,
        offset: int = 0,
    ) -> torch.Tensor:
        if neg_doc_embeddings is None:
            raise ValueError("ColbertListwiseKLLoss requires explicit candidate documents")

        batch_size, num_negs = neg_doc_embeddings.shape[:2]
        if listwise_grades.shape != (batch_size, num_negs + 1):
            raise ValueError(
                "listwise_grades must have shape "
                f"({batch_size}, {num_negs + 1}), got {tuple(listwise_grades.shape)}"
            )

        lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1)
        pos_raw = torch.einsum(
            "bnd,bsd->bns",
            query_embeddings,
            doc_embeddings[offset : offset + batch_size],
        )
        pos_scores = self._aggregate(pos_raw, self.use_smooth_max, dim_max=2, dim_sum=1)
        if self.use_smooth_max:
            neg_raw = torch.einsum("bnd,blsd->blns", query_embeddings, neg_doc_embeddings)
            neg_scores = self._aggregate(neg_raw, True, dim_max=3, dim_sum=2)
        else:
            neg_scores = maxsim_kd(query_embeddings, neg_doc_embeddings)

        if self.normalize_scores:
            pos_scores = self._apply_normalization(pos_scores, lengths)
            neg_scores = self._apply_normalization(neg_scores, lengths)

        student_scores = torch.cat((pos_scores.unsqueeze(1), neg_scores), dim=1)
        student_log_probs = F.log_softmax(student_scores / self.temperature, dim=1)
        teacher_probs = F.softmax(
            listwise_grades.to(dtype=student_scores.dtype) / self.teacher_temperature,
            dim=1,
        )
        loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

        if self.in_batch_term_weight > 0:
            loss_ib = self.inner_loss(query_embeddings, doc_embeddings, offset)
            loss = loss * (1 - self.in_batch_term_weight) + loss_ib * self.in_batch_term_weight
        return loss
