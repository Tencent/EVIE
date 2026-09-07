"""Hierarchical agglomerative clustering (Ward) for late-interaction tokens.

Image tokens only:

* one image per document (``grid_thw`` with t=1);
* L2 semantic vectors fused with same-dimensional 2-D sinusoidal position;
* Ward clustering; each cluster stored as the L2 mean of the raw tokens;
* context tokens stay outside the image span when pooling a full sequence.

No model or dataset dependency.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F


HAC_IMPLEMENTATION_VERSION = (
    "same_dimensional_sinusoidal_2d_ward_context_preserved"
)


@dataclass(frozen=True)
class HACConfig:
    budget: int = 128
    position_weight: float = 0.2
    merge_size: int = 2

    def __post_init__(self) -> None:
        if self.budget <= 0:
            raise ValueError("HAC budget must be positive")
        if not 0.0 <= self.position_weight <= 1.0:
            raise ValueError("HAC position_weight must be in [0, 1]")
        if self.merge_size <= 0:
            raise ValueError("HAC merge_size must be positive")

    @property
    def method(self) -> str:
        return f"hac_k{self.budget}"


def l2_normalize(values: torch.Tensor) -> torch.Tensor:
    """Normalize vectors along the final dimension without changing dtype."""

    if values.ndim == 0:
        raise ValueError("cannot normalize a scalar")
    return values / values.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def position_features(
    grid_thw: list[int] | tuple[int, int, int],
    token_count: int,
    merge_size: int = 2,
) -> torch.Tensor:
    """Return normalized raster-order ``[x, y]`` positions for image tokens."""

    if len(grid_thw) != 3:
        raise ValueError(f"expected [t, h, w] grid, got {grid_thw}")
    temporal, height, width = (int(value) for value in grid_thw)
    if temporal != 1:
        raise ValueError(
            "HAC currently supports one image per document; "
            f"got temporal grid={temporal}"
        )
    if merge_size <= 0:
        raise ValueError("merge_size must be positive")
    if height <= 0 or width <= 0:
        raise ValueError(f"grid height/width must be positive, got {grid_thw}")
    if height % merge_size or width % merge_size:
        raise ValueError(
            f"grid {grid_thw} is not divisible by merge_size={merge_size}"
        )

    grid_height = height // merge_size
    grid_width = width // merge_size
    expected = grid_height * grid_width
    if expected != token_count:
        raise ValueError(
            f"grid mismatch: {grid_height}*{grid_width} != {token_count}; "
            f"grid={list(grid_thw)}, merge_size={merge_size}"
        )
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, grid_height),
        torch.linspace(0.0, 1.0, grid_width),
        indexing="ij",
    )
    return torch.stack((xx.flatten(), yy.flatten()), dim=-1)


def sinusoidal_2d(position: torch.Tensor, dim: int) -> torch.Tensor:
    """Encode normalized 2-D positions into the semantic vector dimension."""

    if position.ndim != 2 or position.shape[-1] != 2:
        raise ValueError(
            f"position must have shape [tokens, 2], got {tuple(position.shape)}"
        )
    if dim <= 0:
        raise ValueError("embedding dimension must be positive")
    quarter = max(1, dim // 4)
    frequencies = torch.exp(
        torch.arange(quarter, dtype=torch.float32)
        * (-math.log(10000.0) / max(1, quarter - 1))
    )
    x = position[:, 0:1].float() / frequencies.view(1, -1)
    y = position[:, 1:2].float() / frequencies.view(1, -1)
    encoded = torch.cat(
        (torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)),
        dim=1,
    )
    if encoded.shape[1] < dim:
        encoded = F.pad(encoded, (0, dim - encoded.shape[1]))
    return l2_normalize(encoded[:, :dim])


def hac_assignment(
    image_embedding: torch.Tensor,
    grid_thw: list[int] | tuple[int, int, int],
    config: HACConfig = HACConfig(),
) -> torch.Tensor:
    """Ward assignment for image tokens in fused semantic+position space."""

    values = torch.as_tensor(image_embedding).float()
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(
            f"image_embedding must be non-empty [tokens, dim], got {tuple(values.shape)}"
        )
    budget = min(config.budget, int(values.shape[0]))
    if budget == values.shape[0]:
        return torch.arange(values.shape[0], dtype=torch.long)

    position = position_features(
        grid_thw,
        int(values.shape[0]),
        config.merge_size,
    )
    position_encoding = sinusoidal_2d(position, int(values.shape[1]))
    fused = l2_normalize(
        (1.0 - config.position_weight) * l2_normalize(values)
        + config.position_weight * position_encoding
    )

    from sklearn.cluster import AgglomerativeClustering

    labels = AgglomerativeClustering(
        n_clusters=budget,
        linkage="ward",
    ).fit_predict(fused.numpy())
    assignment = torch.from_numpy(labels).long()
    counts = torch.bincount(assignment, minlength=budget)
    if (counts == 0).any():
        raise RuntimeError("HAC produced an empty cluster")
    return assignment


def pool_assignment(
    image_embedding: torch.Tensor,
    assignment: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool assigned image tokens and normalize every pooled vector."""

    values = torch.as_tensor(image_embedding).float()
    labels = torch.as_tensor(assignment, dtype=torch.long)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("image_embedding must be non-empty [tokens, dim]")
    if labels.ndim != 1 or labels.shape[0] != values.shape[0]:
        raise ValueError("assignment length must equal the image token count")
    if (labels < 0).any():
        raise ValueError("assignment labels must be non-negative")

    budget = int(labels.max().item()) + 1
    counts = torch.bincount(labels, minlength=budget)
    if (counts == 0).any():
        raise ValueError("assignment contains an empty cluster")
    pooled = torch.stack(
        [
            values[labels == cluster].mean(dim=0)
            for cluster in range(budget)
        ]
    )
    return l2_normalize(pooled)


def hac_pool(
    image_embedding: torch.Tensor,
    grid_thw: list[int] | tuple[int, int, int],
    budget: int = 128,
    position_weight: float = 0.2,
    merge_size: int = 2,
) -> torch.Tensor:
    """Pool image tokens with Ward HAC."""

    config = HACConfig(
        budget=budget,
        position_weight=position_weight,
        merge_size=merge_size,
    )
    values = torch.as_tensor(image_embedding).float()
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("image_embedding must be non-empty [tokens, dim]")
    if config.budget >= values.shape[0]:
        return l2_normalize(values)
    assignment = hac_assignment(values, grid_thw, config)
    pooled = pool_assignment(values, assignment)
    if pooled.shape[0] != min(config.budget, values.shape[0]):
        raise RuntimeError("HAC pooled token count does not match budget")
    return pooled


def image_span_from_mask(image_mask: torch.Tensor) -> tuple[int, int]:
    """Validate a contiguous image mask and return its half-open span."""

    mask = torch.as_tensor(image_mask).bool().flatten()
    positions = torch.nonzero(mask, as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError("image mask is empty")
    expected = torch.arange(
        int(positions[0]),
        int(positions[-1]) + 1,
        dtype=positions.dtype,
    )
    if not torch.equal(positions.cpu(), expected):
        raise ValueError("HAC requires one contiguous image-token span")
    return int(positions[0]), int(positions[-1]) + 1


def hac_pool_full_embedding(
    full_embedding: torch.Tensor,
    grid_thw: list[int] | tuple[int, int, int],
    image_start: int,
    image_end: int,
    budget: int = 128,
    position_weight: float = 0.2,
    merge_size: int = 2,
) -> torch.Tensor:
    """Compress only the image span and preserve context tokens."""

    full = torch.as_tensor(full_embedding).float()
    if full.ndim != 2 or full.shape[0] == 0:
        raise ValueError("full_embedding must be non-empty [tokens, dim]")
    if not 0 <= image_start < image_end <= full.shape[0]:
        raise ValueError(
            f"invalid image span [{image_start}, {image_end}) for "
            f"{full.shape[0]} full tokens"
        )
    pooled = hac_pool(
        full[image_start:image_end],
        grid_thw,
        budget=budget,
        position_weight=position_weight,
        merge_size=merge_size,
    )
    return l2_normalize(
        torch.cat((full[:image_start], pooled, full[image_end:]), dim=0)
    )


def hac_metadata(
    config: HACConfig = HACConfig(),
    **values: Any,
) -> dict[str, Any]:
    """Index metadata fields plus caller statistics."""

    metadata: dict[str, Any] = {
        "method": config.method,
        "budget": config.budget,
        "position_weight": config.position_weight,
        "merge_size": config.merge_size,
        "implementation_version": HAC_IMPLEMENTATION_VERSION,
    }
    metadata.update(values)
    return metadata
