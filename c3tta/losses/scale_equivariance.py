"""Source-only paired scale-canvas views and geometry consistency losses."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F


@dataclass
class ScaleCanvasBatch:
    """A synthetic global view and the exact output-to-input sampling grid."""

    images: torch.Tensor
    seg_target: torch.Tensor | None
    grid: torch.Tensor
    scales: torch.Tensor
    center_offsets: torch.Tensor


def paired_scale_equivariance_config(
    train_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the optional source-only paired geometry objective."""

    raw = train_config.get("paired_scale_equivariance", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("train.paired_scale_equivariance must be a mapping")
    result: dict[str, Any] = {
        "enabled": bool(raw.get("enabled", False)),
        "scale_min": float(raw.get("scale_min", 0.20)),
        "scale_max": float(raw.get("scale_max", 1.00)),
        "log_uniform": bool(raw.get("log_uniform", True)),
        "center_jitter": float(raw.get("center_jitter", 0.20)),
        "global_supervision_weight": float(
            raw.get("global_supervision_weight", 1.0)
        ),
        "seg_consistency_weight": float(raw.get("seg_consistency_weight", 0.0)),
        "cls_consistency_weight": float(raw.get("cls_consistency_weight", 0.0)),
        "clinical_consistency_weight": float(
            raw.get("clinical_consistency_weight", 0.0)
        ),
        "warmup_epochs": int(raw.get("warmup_epochs", 0)),
    }
    if not 0.0 < result["scale_min"] <= result["scale_max"] <= 1.0:
        raise ValueError(
            "paired scale range must satisfy 0 < scale_min <= scale_max <= 1"
        )
    if not 0.0 <= result["center_jitter"] <= 0.5:
        raise ValueError("paired center_jitter must be in [0, 0.5]")
    if result["warmup_epochs"] < 0:
        raise ValueError("paired scale warmup_epochs must be non-negative")
    for name in (
        "global_supervision_weight",
        "seg_consistency_weight",
        "cls_consistency_weight",
        "clinical_consistency_weight",
    ):
        if result[name] < 0.0:
            raise ValueError(f"paired scale {name} must be non-negative")
    if result["enabled"] and result["global_supervision_weight"] <= 0.0:
        raise ValueError(
            "enabled paired scale training requires positive global supervision"
        )
    return result


def _as_batch_vector(
    value: torch.Tensor | None,
    *,
    batch_size: int,
    width: int,
    device: torch.device,
    name: str,
) -> torch.Tensor | None:
    if value is None:
        return None
    result = value.to(device=device, dtype=torch.float32)
    expected = (batch_size,) if width == 1 else (batch_size, width)
    if tuple(result.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(result.shape)}")
    return result


def scale_canvas_batch(
    images: torch.Tensor,
    seg_target: torch.Tensor | None,
    config: Mapping[str, Any],
    *,
    scales: torch.Tensor | None = None,
    center_offsets: torch.Tensor | None = None,
) -> ScaleCanvasBatch:
    """Create a differentiable source-derived global view for each sample.

    ``center_offsets`` are fractions relative to the canvas center, so ``0.2``
    means a requested center at 70% of the corresponding image dimension. The
    realized center is clamped so the scaled source remains fully in-frame,
    matching :func:`c3tta.data.transforms.scale_canvas_pair`.
    """

    if images.ndim != 4:
        raise ValueError("images must have shape [B, C, H, W]")
    if not images.is_floating_point():
        raise ValueError("images must be floating point")
    batch_size = int(images.shape[0])
    if batch_size <= 0:
        raise ValueError("images must contain at least one sample")
    if seg_target is not None:
        if seg_target.ndim != 4 or seg_target.shape[0] != batch_size:
            raise ValueError("seg_target must have shape [B, C, H, W]")
        if seg_target.shape[-2:] != images.shape[-2:]:
            raise ValueError("seg_target and images must have the same spatial size")

    scale_min = float(config["scale_min"])
    scale_max = float(config["scale_max"])
    jitter = float(config["center_jitter"])
    sampled_scales = _as_batch_vector(
        scales,
        batch_size=batch_size,
        width=1,
        device=images.device,
        name="scales",
    )
    if sampled_scales is None:
        unit = torch.rand(batch_size, device=images.device, dtype=torch.float32)
        if bool(config.get("log_uniform", True)):
            sampled_scales = torch.exp(
                math.log(scale_min)
                + unit * (math.log(scale_max) - math.log(scale_min))
            )
        else:
            sampled_scales = scale_min + unit * (scale_max - scale_min)
    if bool(
        torch.any((sampled_scales < scale_min) | (sampled_scales > scale_max)).item()
    ):
        raise ValueError("scales fall outside the configured scale range")

    sampled_offsets = _as_batch_vector(
        center_offsets,
        batch_size=batch_size,
        width=2,
        device=images.device,
        name="center_offsets",
    )
    if sampled_offsets is None:
        sampled_offsets = torch.empty(
            (batch_size, 2), device=images.device, dtype=torch.float32
        ).uniform_(-jitter, jitter)
    if bool(torch.any(sampled_offsets.abs() > jitter + 1e-7).item()):
        raise ValueError("center_offsets fall outside the configured jitter")

    # affine_grid expects an output-to-input map in normalized coordinates.
    # A source scaled by s occupies width 2s in the output, hence x_in=(x-c)/s.
    requested_center = 2.0 * sampled_offsets
    maximum_center = 1.0 - sampled_scales
    realized_center = torch.maximum(
        torch.minimum(requested_center, maximum_center[:, None]),
        -maximum_center[:, None],
    )
    theta = torch.zeros(
        (batch_size, 2, 3), device=images.device, dtype=torch.float32
    )
    inverse_scale = sampled_scales.reciprocal()
    theta[:, 0, 0] = inverse_scale
    theta[:, 1, 1] = inverse_scale
    theta[:, 0, 2] = -realized_center[:, 0] * inverse_scale
    theta[:, 1, 2] = -realized_center[:, 1] * inverse_scale
    grid = F.affine_grid(theta, images.shape, align_corners=False)

    transformed_images = F.grid_sample(
        images.float(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).to(dtype=images.dtype)
    transformed_target = None
    if seg_target is not None:
        transformed_target = F.grid_sample(
            seg_target.float(),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        ).to(dtype=seg_target.dtype)
    return ScaleCanvasBatch(
        transformed_images,
        transformed_target,
        grid,
        sampled_scales,
        realized_center / 2.0,
    )


def _probability_dice_loss(
    prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    dims = tuple(range(2, prediction.ndim))
    intersection = (prediction * target).sum(dim=dims)
    # The squared denominator makes this a true consistency discrepancy:
    # identical soft probability maps have exactly zero loss. The usual
    # mask-supervision Dice denominator (sum p + sum y) would sharpen a soft
    # teacher even when student and teacher already match.
    denominator = prediction.square().sum(dim=dims) + target.square().sum(dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def paired_scale_equivariance_consistency_loss(
    clean_output: Any,
    global_output: Any,
    grid: torch.Tensor,
    *,
    seg_weight: float,
    cls_weight: float,
    clinical_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Anchor the global prediction to an exactly warped clean prediction."""

    clean_probability = torch.sigmoid(clean_output.seg_logits.detach().float())
    expected_global = F.grid_sample(
        clean_probability,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    global_probability = torch.sigmoid(global_output.seg_logits.float())
    seg_equivariance = _probability_dice_loss(global_probability, expected_global)

    clean_cls = torch.sigmoid(clean_output.cls_logits.detach().float())
    global_cls = torch.sigmoid(global_output.cls_logits.float())
    cls_invariance = F.mse_loss(global_cls, clean_cls)

    vcdr_invariance = F.smooth_l1_loss(
        global_output.soft_vcdr.float(), clean_output.soft_vcdr.detach().float()
    )
    rim_invariance = F.smooth_l1_loss(
        global_output.rim_proxy.float(), clean_output.rim_proxy.detach().float()
    )
    clinical_invariance = 0.5 * (vcdr_invariance + rim_invariance)
    graph_zero = (
        global_output.seg_logits.sum() + global_output.cls_logits.sum()
    ).float() * 0.0
    loss = (
        float(seg_weight) * seg_equivariance
        + float(cls_weight) * cls_invariance
        + float(clinical_weight) * clinical_invariance
        + graph_zero
    )
    return loss, {
        "seg_equivariance": float(seg_equivariance.detach()),
        "cls_invariance": float(cls_invariance.detach()),
        "vcdr_invariance": float(vcdr_invariance.detach()),
        "rim_invariance": float(rim_invariance.detach()),
        "clinical_invariance": float(clinical_invariance.detach()),
        "weighted_consistency": float(loss.detach()),
    }
