"""Unified tiny/DINOv2/DINOv3 feature extraction interface."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn


@dataclass
class BackboneOutput:
    features: list[torch.Tensor]
    global_feat: torch.Tensor


class LoRALinear(nn.Module):
    """A frozen Linear layer with a trainable low-rank residual update.

    The base weight remains exactly the published foundation-model weight.  A
    zero-initialized up projection makes injection an identity operation at
    step zero, which is important when adapting a large encoder on the small
    HUC source set.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0.0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.lora_down(self.dropout(x))) * self.scaling


class TinyBackbone(nn.Module):
    def __init__(self, embed_dim: int = 64, levels: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = 16
        blocks = []
        in_ch = 3
        for level in range(levels):
            out_ch = embed_dim
            stride = 2 if level else 4
            blocks.append(nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1), nn.GELU(), nn.GroupNorm(1, out_ch)))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return BackboneOutput(features=features, global_feat=features[-1].mean(dim=(-2, -1)))


class TimmBackbone(nn.Module):
    def __init__(
        self,
        name: str,
        checkpoint: str | None,
        feature_indices: Sequence[int],
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Install timm to use DINO backbones, or use --backbone tiny for smoke tests") from exc
        # Keep pretrained downloads explicit. A local checkpoint is preferred
        # for reproducible/air-gapped experiments; otherwise timm may fetch the
        # published DINOv2 weights.
        self.model = timm.create_model(
            name,
            pretrained=pretrained and checkpoint is None,
            img_size=None,
            dynamic_img_size=True,
        )
        if checkpoint is not None:
            # Official DINOv2 training checkpoints contain ``mask_token``,
            # which is intentionally absent from timm's inference ViT.  The
            # timm loader handles common checkpoint containers/prefixes; a
            # non-strict load preserves all matching inference weights.
            if str(checkpoint).lower().endswith((".pth", ".pt")):
                # RETFound/FunduSegmenter publishes MAE checkpoints in the
                # standard ``{"model": state_dict, ...}`` container.  timm's
                # generic loader does not consistently unwrap that container
                # across versions, so normalize it here while retaining
                # non-strict loading for the discarded MAE decoder.
                payload = torch.load(checkpoint, map_location="cpu")
                state = payload.get("model", payload.get("state_dict", payload))
                if not isinstance(state, dict):
                    raise ValueError(f"Checkpoint does not contain a state dict: {checkpoint}")
                normalized = {}
                for key, value in state.items():
                    name = str(key)
                    for prefix in ("module.", "encoder."):
                        if name.startswith(prefix):
                            name = name[len(prefix) :]
                    if name.startswith("decoder.") or name == "mask_token":
                        continue
                    normalized[name] = value
                incompatible = self.model.load_state_dict(normalized, strict=False)
                if not incompatible.missing_keys:
                    pass
            else:
                from timm.models import load_checkpoint

                load_checkpoint(self.model, checkpoint, strict=False)
        self.feature_indices = tuple(feature_indices)
        self.embed_dim = getattr(self.model, "num_features", None) or getattr(self.model, "embed_dim", None)
        if self.embed_dim is None:
            raise ValueError(f"Cannot infer feature dimension from {name}")
        self.patch_size = getattr(getattr(self.model, "patch_embed", None), "patch_size", 14)
        if isinstance(self.patch_size, tuple):
            self.patch_size = self.patch_size[0]
        self.lora_module_names: list[str] = []

    def enable_lora(
        self,
        rank: int,
        alpha: float,
        last_blocks: int,
        targets: Sequence[str] = ("attn.qkv",),
        dropout: float = 0.0,
    ) -> None:
        """Inject LoRA into the final Transformer blocks after loading weights."""
        if rank <= 0:
            return
        if self.lora_module_names:
            raise RuntimeError("LoRA has already been injected into this backbone")
        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            raise RuntimeError("This timm backbone does not expose Transformer blocks for LoRA")
        block_count = len(blocks)
        if last_blocks <= 0 or last_blocks > block_count:
            raise ValueError(
                f"lora_last_blocks must be in [1, {block_count}], got {last_blocks}"
            )
        if not targets:
            raise ValueError("LoRA needs at least one target Linear layer")
        for block_index in range(block_count - last_blocks, block_count):
            block = blocks[block_index]
            for target in targets:
                parent: nn.Module = block
                components = str(target).split(".")
                for component in components[:-1]:
                    child = getattr(parent, component, None)
                    if not isinstance(child, nn.Module):
                        raise ValueError(
                            f"LoRA target {target!r} was not found in Transformer block {block_index}"
                        )
                    parent = child
                leaf = components[-1]
                linear = getattr(parent, leaf, None)
                if not isinstance(linear, nn.Linear):
                    raise ValueError(
                        f"LoRA target {target!r} in Transformer block {block_index} is not nn.Linear"
                    )
                setattr(parent, leaf, LoRALinear(linear, rank, alpha, dropout))
                self.lora_module_names.append(f"model.blocks.{block_index}.{target}")

    def trainable_lora_parameters(self) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        for module in self.modules():
            if isinstance(module, LoRALinear):
                parameters.extend(
                    parameter
                    for parameter in list(module.lora_down.parameters()) + list(module.lora_up.parameters())
                )
        return parameters

    def _reshape_tokens(self, token: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if token.ndim == 4:
            return token
        if token.ndim != 3:
            raise ValueError(f"Unexpected intermediate shape: {tuple(token.shape)}")
        b, n, c = token.shape
        h = x.shape[-2] // int(self.patch_size)
        w = x.shape[-1] // int(self.patch_size)
        if n == h * w + 1:
            token = token[:, 1:]
        if token.shape[1] != h * w:
            h = w = int(token.shape[1] ** 0.5)
        return token.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        model = self.model
        features = None
        if hasattr(model, "forward_intermediates"):
            try:
                result = model.forward_intermediates(x, indices=self.feature_indices, output_fmt="NCHW", intermediates_only=True, norm=True)
                features = result[0] if isinstance(result, tuple) else result
            except (TypeError, RuntimeError):
                features = None
        if features is None and hasattr(model, "get_intermediate_layers"):
            result = model.get_intermediate_layers(x, n=self.feature_indices, reshape=True, return_prefix_tokens=False, norm=True)
            features = list(result)
        if features is None:
            raise RuntimeError("Backbone does not expose intermediate features")
        features = [self._reshape_tokens(f, x) for f in features]
        return BackboneOutput(features=features, global_feat=features[-1].mean(dim=(-2, -1)))


def build_backbone(
    name: str = "tiny",
    checkpoint: str | None = None,
    feature_indices: Sequence[int] = (2, 5, 8, 11),
    image_size: int = 448,
    pretrained: bool = True,
) -> nn.Module:
    name_lower = name.lower()
    if name_lower == "tiny":
        return TinyBackbone()
    aliases = {
        "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
        "dinov3_vits16": "vit_small_patch16_dinov3.lvd1689m",
        # RETFound uses a ViT-L/16 MAE encoder.  The official checkpoint is
        # supplied separately because its license/access terms differ from
        # the generic DINO weights.
        "retfound": "vit_large_patch16_224",
        "retfound_mae": "vit_large_patch16_224",
        "retfound_mae_vit_large_patch16": "vit_large_patch16_224",
    }
    return TimmBackbone(aliases.get(name_lower, name), checkpoint, feature_indices, pretrained)
