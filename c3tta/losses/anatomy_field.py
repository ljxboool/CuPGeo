"""Scale-normalized OD/OC anatomy-field supervision.

The field target is a signed distance to each OD/OC boundary divided by the
source optic-disc diameter.  Unlike a binary mask, this representation is
dimensionless: an isotropic scale-canvas transform rescales both numerator and
denominator by the same factor.  The module deliberately leaves the ordinary
mask loss untouched and is only enabled inside the paired source-only scale
protocol.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F


def scale_normalized_anatomy_field_config(
    train_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the optional DAFL feasibility objective."""

    raw = train_config.get("scale_normalized_anatomy_field", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("train.scale_normalized_anatomy_field must be a mapping")
    result = {
        "enabled": bool(raw.get("enabled", False)),
        "temperature": float(raw.get("temperature", 0.05)),
        "supervision_weight": float(raw.get("supervision_weight", 0.15)),
        "equivariance_weight": float(raw.get("equivariance_weight", 0.20)),
        "boundary_band": float(raw.get("boundary_band", 0.15)),
        "mask_consistency_weight": float(raw.get("mask_consistency_weight", 0.0)),
        "mask_consistency_temperature": float(raw.get("mask_consistency_temperature", 0.10)),
        "warmup_epochs": int(raw.get("warmup_epochs", 0)),
    }
    if result["temperature"] <= 0.0:
        raise ValueError("anatomy-field temperature must be positive")
    if result["boundary_band"] <= 0.0:
        raise ValueError("anatomy-field boundary_band must be positive")
    if result["mask_consistency_temperature"] <= 0.0:
        raise ValueError("anatomy-field mask_consistency_temperature must be positive")
    if result["warmup_epochs"] < 0:
        raise ValueError("anatomy-field warmup_epochs must be non-negative")
    for name in (
        "supervision_weight",
        "equivariance_weight",
        "mask_consistency_weight",
    ):
        if result[name] < 0.0:
            raise ValueError(f"anatomy-field {name} must be non-negative")
    if result["enabled"] and (
        result["supervision_weight"] + result["equivariance_weight"] <= 0.0
    ):
        raise ValueError("enabled anatomy-field supervision needs a positive loss weight")
    return result


def warp_dimensionless_field(field: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Warp a [-1, 1] field, preserving the correct negative exterior padding."""

    if field.ndim != 4:
        raise ValueError("field must have shape [B, C, H, W]")
    shifted = 0.5 * (field.float() + 1.0)
    warped = F.grid_sample(
        shifted,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return 2.0 * warped - 1.0


def _boundary_weight(target: torch.Tensor, band: float) -> torch.Tensor:
    """Emphasize the informative field neighborhood without ignoring interiors."""

    return 0.15 + 0.85 * torch.exp(-target.abs() / float(band))


def anatomy_field_regression_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    boundary_band: float,
) -> torch.Tensor:
    """Robustly fit a clipped continuous anatomy field."""

    if logits.shape != target.shape:
        raise ValueError(
            f"field logit/target shape mismatch: {tuple(logits.shape)} vs {tuple(target.shape)}"
        )
    prediction = torch.tanh(logits.float())
    target = target.float()
    weight = _boundary_weight(target, boundary_band)
    per_pixel = F.smooth_l1_loss(prediction, target, beta=0.15, reduction="none")
    return (per_pixel * weight).sum() / weight.sum().clamp_min(1e-6)


def field_mask_consistency_loss(
    seg_logits: torch.Tensor,
    field_logits: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Align an OD/OC mask with the signed field's inside/outside semantics.

    The stored field is positive inside an anatomical structure and negative
    outside it.  A temperature-scaled sigmoid therefore maps it to the same
    occupancy convention as the segmentation branch.
    """

    if seg_logits.shape != field_logits.shape:
        raise ValueError(
            "segmentation and anatomy-field logits must share shape, got "
            f"{tuple(seg_logits.shape)} vs {tuple(field_logits.shape)}"
        )
    if temperature <= 0.0:
        raise ValueError("field/mask consistency temperature must be positive")
    mask_probability = torch.sigmoid(seg_logits.float())
    field_probability = torch.sigmoid(field_logits.float() / float(temperature))
    return F.smooth_l1_loss(field_probability, mask_probability, beta=0.10)


def paired_scale_anatomy_field_loss(
    clean_field_logits: torch.Tensor,
    global_field_logits: torch.Tensor,
    clean_target: torch.Tensor,
    grid: torch.Tensor,
    *,
    supervision_weight: float,
    equivariance_weight: float,
    boundary_band: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervise the field on both views and enforce dimensionless consistency.

    The scaled target is the bilinearly warped clean continuous target rather
    than a nearest-neighbor transformed mask.  This is the key operational
    distinction from ordinary scale-canvas segmentation supervision.
    """

    global_target = warp_dimensionless_field(clean_target, grid)
    clean_supervised = anatomy_field_regression_loss(
        clean_field_logits, clean_target, boundary_band=boundary_band
    )
    global_supervised = anatomy_field_regression_loss(
        global_field_logits, global_target, boundary_band=boundary_band
    )
    expected_global = warp_dimensionless_field(
        torch.tanh(clean_field_logits.detach().float()), grid
    )
    equivariance = anatomy_field_regression_loss(
        global_field_logits, expected_global, boundary_band=boundary_band
    )
    supervised_mean = 0.5 * (clean_supervised + global_supervised)
    loss = float(supervision_weight) * supervised_mean + float(
        equivariance_weight
    ) * equivariance
    return loss, {
        "clean_field_supervised": float(clean_supervised.detach()),
        "global_field_supervised": float(global_supervised.detach()),
        "field_equivariance": float(equivariance.detach()),
        "weighted_field": float(loss.detach()),
    }
