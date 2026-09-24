"""Topology objectives for nested optic-disc/optic-cup segmentation."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F


def soft_topology_config(train_config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the optional threshold-aware OD/OC topology objective."""

    raw = train_config.get("soft_topology", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("train.soft_topology must be a mapping")
    gradient_target = (
        str(raw.get("gradient_target", "disc_only"))
        .strip()
        .lower()
        .replace("-", "_")
    )
    result: dict[str, Any] = {
        "enabled": bool(raw.get("enabled", False)),
        "weight": float(raw.get("weight", 0.05)),
        "start_epoch": int(raw.get("start_epoch", 0)),
        "ramp_epochs": int(raw.get("ramp_epochs", 0)),
        "threshold": float(raw.get("threshold", 0.5)),
        "topk_fraction": float(raw.get("topk_fraction", 0.01)),
        "gradient_target": gradient_target,
    }
    if result["weight"] < 0.0:
        raise ValueError("soft-topology weight must be non-negative")
    if result["start_epoch"] < 0:
        raise ValueError("soft-topology start_epoch must be non-negative")
    if result["ramp_epochs"] < 0:
        raise ValueError("soft-topology ramp_epochs must be non-negative")
    if not 0.0 < result["threshold"] < 1.0:
        raise ValueError("soft-topology threshold must be in (0, 1)")
    if not 0.0 < result["topk_fraction"] <= 1.0:
        raise ValueError("soft-topology topk_fraction must be in (0, 1]")
    if gradient_target not in {"both", "disc_only"}:
        raise ValueError(
            "soft-topology gradient_target must be 'both' or 'disc_only'"
        )
    return result


def thresholded_topology_loss(
    seg_logits: torch.Tensor,
    *,
    threshold: float,
    topk_fraction: float,
    gradient_target: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Penalize only OC-positive/OD-negative pixels at the decision threshold.

    Unlike a global probability-order penalty, this objective is exactly zero
    when the binary masks are already nested.  In particular, it does not
    alter a legal high-vCDR pixel merely because both channels are foreground
    and ``p_OC > p_OD``.  Top-k aggregation prevents a thin violation band from
    being diluted by the full image area.  The denominator remains the fixed
    configured top-k area; ``active_topk_fraction`` makes any sparse-violation
    dilution explicit in the training log.

    ``disc_only`` stops the topology gradient through OC and repairs a
    violation by increasing OD support.  This protects the cup boundary from
    the attenuation observed with the legacy ``p_OC = p_OD * q_OC`` model.
    Ordinary supervised OD/OC losses remain responsible for rejecting false
    cup positives and limiting unnecessary disc expansion.
    """

    if seg_logits.ndim != 4 or seg_logits.shape[1] != 2:
        raise ValueError("OD/OC logits must have shape [B, 2, H, W]")
    if not seg_logits.is_floating_point():
        raise ValueError("OD/OC logits must be floating point")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    if not 0.0 < float(topk_fraction) <= 1.0:
        raise ValueError("topk_fraction must be in (0, 1]")
    if gradient_target not in {"both", "disc_only"}:
        raise ValueError("gradient_target must be 'both' or 'disc_only'")

    probabilities = torch.sigmoid(seg_logits.float())
    od_probability = probabilities[:, 0:1]
    oc_probability = probabilities[:, 1:2]
    violation_gate = (
        (oc_probability >= float(threshold))
        & (od_probability < float(threshold))
    ).detach()

    if gradient_target == "disc_only":
        signed_excess = oc_probability.detach() - od_probability
    else:
        signed_excess = oc_probability - od_probability
    severity = F.relu(signed_excess) * violation_gate.to(dtype=probabilities.dtype)

    flattened = severity.flatten(1)
    pixel_count = int(flattened.shape[1])
    topk_count = max(1, min(pixel_count, math.ceil(pixel_count * topk_fraction)))
    per_image_topk = flattened.topk(topk_count, dim=1).values.mean(dim=1)
    loss = per_image_topk.mean()

    with torch.no_grad():
        value_excess = F.relu(oc_probability - od_probability) * violation_gate
        violating_pixels = violation_gate.flatten(1)
        active_topk_fraction = (
            violating_pixels.sum(dim=1).clamp(max=topk_count).float()
            / float(topk_count)
        )
        per_image_conditional_excess = value_excess.flatten(1).sum(dim=1) / (
            violating_pixels.sum(dim=1).clamp_min(1)
        )
        stats = {
            "topk_probability_excess": float(loss.detach()),
            "mean_violating_probability_excess": float(
                per_image_conditional_excess.mean()
            ),
            "violating_pixel_fraction": float(violation_gate.float().mean()),
            "violating_image_fraction": float(
                violating_pixels.any(dim=1).float().mean()
            ),
            "active_topk_fraction": float(active_topk_fraction.mean()),
        }
    return loss, stats
