"""Source-only paired camera-style counterfactual utilities."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

from c3tta.data.transforms import IMAGENET_MEAN, IMAGENET_STD


def paired_counterfactual_config(train_config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the optional source-only paired-view training block."""
    raw = train_config.get("paired_counterfactual", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("train.paired_counterfactual must be a mapping")
    result: dict[str, Any] = {
        "enabled": bool(raw.get("enabled", False)),
        "style_supervision_weight": float(raw.get("style_supervision_weight", 1.0)),
        "seg_consistency_weight": float(raw.get("seg_consistency_weight", 0.0)),
        "cls_consistency_weight": float(raw.get("cls_consistency_weight", 0.0)),
        "boundary_consistency_weight": float(
            raw.get("boundary_consistency_weight", 0.0)
        ),
        "clinical_consistency_weight": float(
            raw.get("clinical_consistency_weight", 0.0)
        ),
    }
    for key in (
        "style_supervision_weight",
        "seg_consistency_weight",
        "cls_consistency_weight",
        "boundary_consistency_weight",
        "clinical_consistency_weight",
    ):
        if result[key] < 0.0:
            raise ValueError(f"paired counterfactual {key} must be non-negative")
    style = raw.get("style", {})
    if not isinstance(style, Mapping):
        raise ValueError("train.paired_counterfactual.style must be a mapping")
    ranges = {
        "channel_gain": (0.84, 1.16),
        "gamma": (0.82, 1.20),
        "brightness": (0.90, 1.10),
        "contrast": (0.90, 1.10),
    }
    style_config: dict[str, float] = {}
    for name, default in ranges.items():
        minimum = float(style.get(f"{name}_min", default[0]))
        maximum = float(style.get(f"{name}_max", default[1]))
        if not 0.0 < minimum <= maximum:
            raise ValueError(
                f"paired counterfactual {name} range must satisfy 0 < min <= max"
            )
        style_config[f"{name}_min"] = minimum
        style_config[f"{name}_max"] = maximum
    for name, default in (("noise_std", 0.0), ("row_noise_std", 0.0)):
        value = float(style.get(name, default))
        if value < 0.0:
            raise ValueError(f"paired counterfactual {name} must be non-negative")
        style_config[name] = value
    result["style"] = style_config
    return result


def _uniform_per_sample(
    batch_size: int,
    channels: int,
    minimum: float,
    maximum: float,
    reference: torch.Tensor,
) -> torch.Tensor:
    shape = (batch_size, channels, 1, 1)
    return torch.empty(shape, device=reference.device, dtype=torch.float32).uniform_(
        minimum, maximum
    )


def camera_style_counterfactual(
    normalized_images: torch.Tensor, style_config: Mapping[str, float]
) -> torch.Tensor:
    """Apply a geometry-preserving camera response intervention in raw RGB space."""
    if normalized_images.ndim != 4 or normalized_images.shape[1] != 3:
        raise ValueError("normalized_images must have shape [B, 3, H, W]")
    images = normalized_images.float()
    mean = IMAGENET_MEAN.to(device=images.device, dtype=images.dtype).unsqueeze(0)
    std = IMAGENET_STD.to(device=images.device, dtype=images.dtype).unsqueeze(0)
    raw = (images * std + mean).clamp(0.0, 1.0)
    batch_size = raw.shape[0]

    gains = _uniform_per_sample(
        batch_size,
        3,
        float(style_config["channel_gain_min"]),
        float(style_config["channel_gain_max"]),
        raw,
    )
    raw = raw * gains
    brightness = _uniform_per_sample(
        batch_size,
        1,
        float(style_config["brightness_min"]),
        float(style_config["brightness_max"]),
        raw,
    )
    raw = raw * brightness
    contrast = _uniform_per_sample(
        batch_size,
        1,
        float(style_config["contrast_min"]),
        float(style_config["contrast_max"]),
        raw,
    )
    spatial_mean = raw.mean(dim=(-2, -1), keepdim=True)
    raw = (raw - spatial_mean) * contrast + spatial_mean
    gamma = _uniform_per_sample(
        batch_size,
        1,
        float(style_config["gamma_min"]),
        float(style_config["gamma_max"]),
        raw,
    )
    raw = raw.clamp(0.0, 1.0).pow(gamma)

    noise_std = float(style_config.get("noise_std", 0.0))
    if noise_std:
        raw = raw + torch.randn_like(raw, dtype=torch.float32) * noise_std
    row_noise_std = float(style_config.get("row_noise_std", 0.0))
    if row_noise_std:
        row_noise = torch.randn(
            (batch_size, 1, raw.shape[-2], 1),
            device=raw.device,
            dtype=torch.float32,
        ) * row_noise_std
        raw = raw + row_noise
    raw = raw.clamp(0.0, 1.0)
    return ((raw - mean) / std).to(dtype=normalized_images.dtype)


def paired_counterfactual_consistency_loss(
    clean_output: Any,
    style_output: Any,
    seg_weight: float,
    cls_weight: float,
    boundary_weight: float = 0.0,
    clinical_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Anchor a styled source view to stop-gradient clean anatomy/task outputs."""
    clean_seg = torch.sigmoid(clean_output.seg_logits.detach().float())
    style_seg = torch.sigmoid(style_output.seg_logits.float())
    clean_cls = torch.sigmoid(clean_output.cls_logits.detach().float())
    style_cls = torch.sigmoid(style_output.cls_logits.float())
    seg_consistency = F.mse_loss(style_seg, clean_seg)
    cls_consistency = F.mse_loss(style_cls, clean_cls)

    clean_boundary_logits = getattr(clean_output, "boundary_logits", None)
    style_boundary_logits = getattr(style_output, "boundary_logits", None)
    if clean_boundary_logits is None or style_boundary_logits is None:
        if float(boundary_weight) > 0.0:
            raise ValueError(
                "boundary consistency requires clean/style boundary_logits"
            )
        boundary_consistency = style_output.seg_logits.sum().float() * 0.0
    else:
        clean_boundary = torch.sigmoid(clean_boundary_logits.detach().float())
        style_boundary = torch.sigmoid(style_boundary_logits.float())
        boundary_consistency = F.mse_loss(style_boundary, clean_boundary)

    clean_vcdr = getattr(clean_output, "soft_vcdr", None)
    style_vcdr = getattr(style_output, "soft_vcdr", None)
    clean_rim = getattr(clean_output, "rim_proxy", None)
    style_rim = getattr(style_output, "rim_proxy", None)
    if any(value is None for value in (clean_vcdr, style_vcdr, clean_rim, style_rim)):
        if float(clinical_weight) > 0.0:
            raise ValueError(
                "clinical consistency requires clean/style soft_vcdr and rim_proxy"
            )
        vcdr_consistency = style_output.seg_logits.sum().float() * 0.0
        rim_consistency = style_output.seg_logits.sum().float() * 0.0
    else:
        vcdr_consistency = F.smooth_l1_loss(
            style_vcdr.float(), clean_vcdr.detach().float()
        )
        rim_consistency = F.smooth_l1_loss(
            style_rim.float(), clean_rim.detach().float()
        )
    clinical_consistency = 0.5 * (vcdr_consistency + rim_consistency)
    graph_zero = (
        style_output.seg_logits.sum() + style_output.cls_logits.sum()
    ) * 0.0
    loss = (
        float(seg_weight) * seg_consistency
        + float(cls_weight) * cls_consistency
        + float(boundary_weight) * boundary_consistency
        + float(clinical_weight) * clinical_consistency
        + graph_zero
    )
    return loss, {
        "seg_consistency": float(seg_consistency.detach()),
        "cls_consistency": float(cls_consistency.detach()),
        "boundary_consistency": float(boundary_consistency.detach()),
        "vcdr_consistency": float(vcdr_consistency.detach()),
        "rim_consistency": float(rim_consistency.detach()),
        "clinical_consistency": float(clinical_consistency.detach()),
        "weighted_consistency": float(loss.detach()),
    }
