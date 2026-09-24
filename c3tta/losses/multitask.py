from __future__ import annotations

import torch
import torch.nn.functional as F


def vertical_geometry_loss(
    logits: torch.Tensor,
    target_lengths: torch.Tensor,
    *,
    log_margin_weight: float = 0.5,
) -> torch.Tensor:
    """Supervise the five vertical allocations and emphasize thin rims.

    The target is a probability vector for [outside-top, upper-rim, cup,
    lower-rim, outside-bottom].  Combining categorical allocation loss with a
    log-length regression term prevents a large cup from satisfying the loss
    while silently erasing a thin residual rim.
    """

    if logits.ndim != 2 or logits.shape[1] != 5:
        raise ValueError("vertical geometry logits must have shape [B, 5]")
    if target_lengths.shape != logits.shape:
        raise ValueError(
            "vertical geometry target must match logits, got "
            f"{tuple(target_lengths.shape)} vs {tuple(logits.shape)}"
        )
    target = target_lengths.float().clamp_min(1e-5)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)
    log_probability = F.log_softmax(logits.float(), dim=1)
    categorical = -(target * log_probability).sum(dim=1).mean()
    predicted = log_probability.exp()
    log_margin = F.smooth_l1_loss(
        torch.log(predicted.clamp_min(1e-5)),
        torch.log(target),
        beta=0.25,
    )
    weight = float(log_margin_weight)
    if weight < 0.0:
        raise ValueError("log_margin_weight must be non-negative")
    return categorical + weight * log_margin


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = tuple(range(2, prob.ndim))
    intersection = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def soft_vertical_diameter(mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return a differentiable normalized vertical diameter for a mask."""
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must have shape [B, 1, H, W]")
    _, _, height, _ = mask.shape
    y = torch.linspace(-1.0, 1.0, height, device=mask.device, dtype=mask.dtype)
    y = y.view(1, 1, height, 1)
    mass_y = mask.sum(dim=3)
    mass = mass_y.sum(dim=2).clamp_min(eps)
    mean = (mass_y * y.squeeze(-1)).sum(dim=2) / mass
    variance = (mass_y * (y.squeeze(-1) - mean.unsqueeze(-1)) ** 2).sum(dim=2) / mass
    return 4.0 * torch.sqrt(variance.clamp_min(eps))


def soft_vcdr_from_masks(seg_masks: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute a differentiable vCDR target from source OD/OC masks."""
    if seg_masks.ndim != 4 or seg_masks.shape[1] != 2:
        raise ValueError("seg_masks must have shape [B, 2, H, W]")
    masks = seg_masks.float().clamp(0.0, 1.0)
    od_diameter = soft_vertical_diameter(masks[:, 0:1], eps=eps)
    oc_diameter = soft_vertical_diameter(masks[:, 1:2], eps=eps)
    return oc_diameter / od_diameter.clamp_min(eps)


def soft_vcdr_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float = 0.05,
) -> torch.Tensor:
    """Robust source-supervised loss aligned with the reported vCDR MAE."""
    predicted = prediction.float().reshape(-1)
    expected = target.float().reshape(-1)
    if predicted.shape != expected.shape:
        raise ValueError(
            "vCDR prediction and target must have the same number of samples, got "
            f"{tuple(predicted.shape)} vs {tuple(expected.shape)}"
        )
    valid = torch.isfinite(predicted) & torch.isfinite(expected)
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    return F.smooth_l1_loss(predicted[valid], expected[valid], beta=beta)


def nesting_penalty(seg_logits: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(seg_logits)
    return torch.relu(prob[:, 1:2] - prob[:, 0:1]).mean()


def boundary_target(seg_target: torch.Tensor) -> torch.Tensor:
    """Build a one-channel OD/OC contour target from binary masks."""
    if seg_target.ndim != 4 or seg_target.shape[1] != 2:
        raise ValueError("seg_target must have shape [B, 2, H, W]")
    target = seg_target.float().clamp(0.0, 1.0)
    dilated = F.max_pool2d(target, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-target, kernel_size=3, stride=1, padding=1)
    contour = (dilated - eroded).amax(dim=1, keepdim=True)
    return contour.clamp(0.0, 1.0)


def boundary_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss for the sparse contour map."""
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(2, 3))
    denominator = probability.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def source_loss(
    seg_logits: torch.Tensor,
    cls_logits: torch.Tensor,
    seg_target: torch.Tensor | None,
    cls_target: torch.Tensor | None,
    seg_bce_weight: float = 1.0,
    seg_dice_weight: float = 1.0,
    cls_weight: float = 1.0,
    nest_weight: float = 0.0,
    boundary_logits: torch.Tensor | None = None,
    boundary_bce_weight: float = 0.0,
    boundary_dice_weight: float = 0.0,
    soft_vcdr: torch.Tensor | None = None,
    vcdr_target: torch.Tensor | None = None,
    vcdr_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    zero = seg_logits.sum() * 0.0
    loss = zero
    stats: dict[str, float] = {}
    if seg_target is not None:
        seg_bce = F.binary_cross_entropy_with_logits(seg_logits, seg_target)
        seg_dice = soft_dice_loss(seg_logits, seg_target)
        loss = loss + seg_bce_weight * seg_bce + seg_dice_weight * seg_dice
        stats.update(seg_bce=float(seg_bce.detach()), seg_dice=float(seg_dice.detach()))
        if nest_weight:
            nest = nesting_penalty(seg_logits)
            loss = loss + nest_weight * nest
            stats["nest"] = float(nest.detach())
        if boundary_bce_weight or boundary_dice_weight:
            if boundary_logits is None:
                raise ValueError(
                    "boundary loss is enabled but the model did not return boundary_logits"
                )
            target_boundary = boundary_target(seg_target).to(boundary_logits.device)
            if boundary_logits.shape != target_boundary.shape:
                raise ValueError(
                    "boundary logits and target must have the same shape, got "
                    f"{tuple(boundary_logits.shape)} vs {tuple(target_boundary.shape)}"
                )
            boundary_bce = F.binary_cross_entropy_with_logits(
                boundary_logits, target_boundary
            )
            boundary_dice = boundary_dice_loss(boundary_logits, target_boundary)
            loss = loss + boundary_bce_weight * boundary_bce + boundary_dice_weight * boundary_dice
            stats.update(
                boundary_bce=float(boundary_bce.detach()),
                boundary_dice=float(boundary_dice.detach()),
            )
        if vcdr_weight:
            if soft_vcdr is None:
                raise ValueError("vCDR loss is enabled but the model returned no soft_vcdr")
            target = (
                soft_vcdr_from_masks(seg_target)
                if vcdr_target is None
                else vcdr_target
            )
            vcdr = soft_vcdr_loss(soft_vcdr, target)
            loss = loss + vcdr_weight * vcdr
            stats["vcdr"] = float(vcdr.detach())
    if cls_target is not None:
        cls = F.binary_cross_entropy_with_logits(cls_logits, cls_target)
        loss = loss + cls_weight * cls
        stats["cls"] = float(cls.detach())
    stats["total"] = float(loss.detach())
    return loss, stats
