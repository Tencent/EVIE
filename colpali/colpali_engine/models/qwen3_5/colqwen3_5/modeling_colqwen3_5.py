from typing import ClassVar, Dict, Optional, Sequence, Union

import torch
from torch import nn
from transformers.models.qwen3_5 import Qwen3_5Config, Qwen3_5Model

# Prefix-MRL defaults: one max-width projection, channel prefixes, per-token L2.
DEFAULT_HEAD_DIMS: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048)


def head_key(dim: int) -> str:
    """ModuleDict key for one head. Keys must be strings, so ``2048`` -> ``d2048``."""
    return f"d{int(dim)}"


def normalize_head_dims(dims: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted({int(d) for d in dims}))


def enable_bidirectional_attention(model: "ColQwen3_5") -> None:
    """
    Encoder-ize full-attention layers for retrieval (Nemotron ColEmbed V2 §3.4.1).

    Qwen3.5 mixes GatedDeltaNet (linear_attention) and full attention. We only flip
    full-attention layers: set ``config.is_causal=False`` so ``create_causal_mask``
    falls back to ``create_bidirectional_mask``, and clear ``is_causal`` on
    ``Qwen3_5Attention`` modules. Linear-attention / recurrent layers are unchanged.
    """
    # Top-level and nested text configs (transformers may nest text_config)
    for cfg in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
        if cfg is not None and hasattr(cfg, "is_causal"):
            cfg.is_causal = False
        elif cfg is not None:
            setattr(cfg, "is_causal", False)

    for module in model.modules():
        # Only touch dense attention blocks; leave linear/recurrent alone
        cls_name = module.__class__.__name__
        if cls_name in ("Qwen3_5Attention", "Qwen3Attention") and hasattr(module, "is_causal"):
            module.is_causal = False


def unwrap_col_model(model: nn.Module, _depth: int = 0) -> "ColQwen3_5":
    """Reach the ``ColQwen3_5`` under PEFT / DDP / ``torch.compile`` wrappers."""
    if isinstance(model, ColQwen3_5):
        return model
    if _depth >= 8:
        raise TypeError("no ColQwen3_5 found (wrapper nesting too deep)")
    for attr in ("module", "base_model", "model", "_orig_mod"):
        inner = getattr(model, attr, None)
        if isinstance(inner, nn.Module) and inner is not model:
            try:
                return unwrap_col_model(inner, _depth + 1)
            except TypeError:
                continue
    raise TypeError(f"no ColQwen3_5 under {type(model).__name__}")


def set_active_head(model: nn.Module, dim: Optional[int]) -> None:
    """Select which head ``forward`` returns; ``None`` returns every head."""
    unwrap_col_model(model).set_active_head(dim)


class ColQwen3_5(Qwen3_5Model):  # noqa: N801
    """
    ColQwen3.5 model implementation, following the architecture from the article "ColPali: Efficient Document Retrieval
    with Vision Language Models" paper. Based on the Qwen3.5 backbone.

    Three projection layouts, chosen by the config:

    * ``config.dim`` only (single projection): one ``Linear(hidden, dim)``.
    * ``config.head_dims`` + ``config.mrl_prefix``: one
      ``Linear(hidden, max(head_dims))``; each width is a normalized prefix.
    * legacy ``config.head_dims``: a ``ModuleDict`` of independent heads, kept
      so older multi-head checkpoints remain inspectable.

    Args:
        config (Qwen3_5Config): The model configuration.
        mask_non_image_embeddings (Optional[bool]): Whether to ignore all tokens embeddings
            except those of the image at inference.
            Defaults to False --> Do not mask any embeddings during forward pass.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    _checkpoint_conversion_mapping = {
        r"^base_model\.model\.custom_text_proj": "custom_text_proj",
    }

    def __init__(self, config: Qwen3_5Config, mask_non_image_embeddings: bool = False):
        super().__init__(config=config)

        hidden_size = getattr(self.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.config, "text_config"):
            hidden_size = getattr(self.config.text_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(f"Unable to determine text hidden size for {type(self.config).__name__}.")
        self.hidden_size_for_heads = int(hidden_size)

        head_dims = getattr(config, "head_dims", None)
        self.mrl_prefix = bool(getattr(config, "mrl_prefix", False))
        if head_dims:
            dims = normalize_head_dims(head_dims)
            if not dims or dims[0] <= 0:
                raise ValueError(f"head_dims must all be positive, got {head_dims}")
            too_wide = [d for d in dims if d > hidden_size]
            if too_wide:
                raise ValueError(
                    f"head dims {too_wide} exceed backbone hidden size {hidden_size}; "
                    "projecting wider than the hidden state cannot add rank"
                )
            self.head_dims: Optional[tuple[int, ...]] = dims
            if self.mrl_prefix:
                self.custom_text_proj = nn.Linear(hidden_size, dims[-1])
            else:
                self.custom_text_proj = nn.ModuleDict(
                    {head_key(d): nn.Linear(hidden_size, d) for d in dims}
                )
            self.dim = dims[-1]
        else:
            self.head_dims = None
            self.mrl_prefix = False
            self.dim = getattr(config, "dim", 128)
            if self.dim > hidden_size:
                raise ValueError(
                    f"head dims [{self.dim}] exceed backbone hidden size {hidden_size}; "
                    "projecting wider than the hidden state cannot add rank"
                )
            self.custom_text_proj = nn.Linear(hidden_size, self.dim)

        self.active_head: Optional[int] = None
        self.padding_side = "left"
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.post_init()

    @property
    def is_matryoshka(self) -> bool:
        return self.head_dims is not None

    def head_module_names(self) -> list[str]:
        """Names for PEFT ``modules_to_save``.

        A ``ModuleDict`` has no ``forward``, so PEFT cannot wrap it as one unit;
        each ``Linear`` under it has to be named individually.
        """
        if self.head_dims is None or self.mrl_prefix:
            return ["custom_text_proj"]
        return [f"custom_text_proj.{head_key(d)}" for d in self.head_dims]

    def enable_prefix_mrl(
        self,
        dims: Sequence[int],
        anchor_dim: Optional[int] = None,
    ) -> None:
        """Expand a loaded single-head checkpoint into one prefix-MRL projection.

        Training deliberately calls this *after* ``from_pretrained`` so the
        deployed Preview d128 projection can be copied exactly into the first 128
        rows. Adapter/full-checkpoint loading instead builds the final shape from
        ``config.mrl_prefix`` and restores the trained projection weights.
        """
        normalized = normalize_head_dims(dims)
        if not normalized or normalized[0] <= 0:
            raise ValueError(f"dims must be positive, got {dims}")
        if normalized[-1] > self.hidden_size_for_heads:
            raise ValueError(
                f"max MRL dim {normalized[-1]} exceeds hidden size {self.hidden_size_for_heads}"
            )
        if not isinstance(self.custom_text_proj, nn.Linear) or self.head_dims is not None:
            raise TypeError("enable_prefix_mrl requires a loaded single-head Linear checkpoint")

        old = self.custom_text_proj
        copied = int(anchor_dim if anchor_dim is not None else old.out_features)
        if copied != old.out_features:
            raise ValueError(
                f"anchor_dim={copied} must match loaded head width {old.out_features}"
            )
        if copied not in normalized:
            raise ValueError(f"anchor_dim={copied} must be one of MRL dims {normalized}")

        wide = nn.Linear(
            self.hidden_size_for_heads,
            normalized[-1],
            bias=old.bias is not None,
            device=old.weight.device,
            dtype=old.weight.dtype,
        )
        self._init_weights(wide)
        with torch.no_grad():
            wide.weight[:copied].copy_(old.weight)
            if wide.bias is not None and old.bias is not None:
                wide.bias[:copied].copy_(old.bias)

        self.custom_text_proj = wide
        self.head_dims = normalized
        self.mrl_prefix = True
        self.dim = normalized[-1]
        self.active_head = None
        self.config.head_dims = list(normalized)
        self.config.mrl_prefix = True
        self.config.anchor_dim = copied
        self.config.dim = self.dim

    def set_active_head(self, dim: Optional[int]) -> None:
        if self.head_dims is None:
            if dim is not None and int(dim) != int(self.dim):
                raise ValueError(
                    f"single-head model has dim={self.dim}; cannot select head {dim}"
                )
            return
        if dim is None:
            self.active_head = None
            return
        chosen = int(dim)
        if chosen not in self.head_dims:
            raise ValueError(f"head {chosen} not in {self.head_dims}")
        self.active_head = chosen

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        key_mapping = kwargs.pop("key_mapping", None)
        if key_mapping is None:
            key_mapping = dict(getattr(super(), "_checkpoint_conversion_mapping", {}))
            key_mapping.update(cls._checkpoint_conversion_mapping)
        return super().from_pretrained(*args, **kwargs, key_mapping=key_mapping)

    def enable_bidirectional_attention(self) -> None:
        """See module-level ``enable_bidirectional_attention``."""
        enable_bidirectional_attention(self)

    def _normalize_projected(self, proj: torch.Tensor, kwargs: dict) -> torch.Tensor:
        # Per-token L2 normalization. clamp() guards the narrow heads: MaxSim needs
        # pad positions to be exactly zero, and 0/0 would poison the whole row.
        proj = proj / proj.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)

        if "pixel_values" in kwargs and self.mask_non_image_embeddings:
            # Pools only the image embeddings
            image_mask = (kwargs["input_ids"] == self.config.image_token_id).unsqueeze(-1)
            proj = proj * image_mask
        return proj

    def forward(self, *args, **kwargs) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        # Handle the custom "pixel_values" input obtained with `ColQwen3_5Processor` through unpadding
        if "pixel_values" in kwargs:
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]  # (batch_size,)
            kwargs["pixel_values"] = torch.cat(
                [pixel_sequence[:offset] for pixel_sequence, offset in zip(kwargs["pixel_values"], offsets)],
                dim=0,
            )

        kwargs.pop("return_dict", True)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)
        last_hidden_states = (
            super()
            .forward(*args, **kwargs, use_cache=False, output_hidden_states=True, return_dict=True)
            .last_hidden_state
        )  # (batch_size, sequence_length, hidden_size)

        if self.head_dims is None:
            return self._normalize_projected(self.custom_text_proj(last_hidden_states), kwargs)

        if self.mrl_prefix:
            full = self.custom_text_proj(last_hidden_states)
            if self.active_head is not None:
                return self._normalize_projected(full[..., : self.active_head], kwargs)
            return {
                d: self._normalize_projected(full[..., :d], kwargs)
                for d in self.head_dims
            }

        if self.active_head is not None:
            return self._normalize_projected(
                self.custom_text_proj[head_key(self.active_head)](last_hidden_states), kwargs
            )

        return {
            d: self._normalize_projected(
                self.custom_text_proj[head_key(d)](last_hidden_states), kwargs
            )
            for d in self.head_dims
        }

    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size
