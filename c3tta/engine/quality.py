"""Label-free image-quality measurements used by deployment-time routing."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter
import torch


def gray_gaussian_residual_std(image: Image.Image, blur_radius: float = 1.0) -> float:
    """Return a deterministic high-frequency residual score in raw [0, 1] space.

    The score is intentionally model-free: it measures the standard deviation
    of grayscale image detail remaining after a small Gaussian blur.  It can
    route unusually noisy images without importing labels or adapting model
    parameters.  It is not a diagnostic score and must not be used as one.
    """
    if blur_radius <= 0.0:
        raise ValueError("blur_radius must be positive")
    grayscale = image.convert("L")
    raw = np.asarray(grayscale, dtype=np.float32) / 255.0
    blurred = np.asarray(
        grayscale.filter(ImageFilter.GaussianBlur(radius=float(blur_radius))),
        dtype=np.float32,
    ) / 255.0
    return float(np.std(raw - blurred))


def gray_gaussian_residual_ratio(
    image: Image.Image,
    blur_radius: float = 1.0,
    eps: float = 1e-6,
) -> float:
    """Return high-frequency detail normalized by the image's global contrast.

    Raw high-pass energy becomes small for both blur and a dim low-contrast
    photograph.  Dividing by the grayscale standard deviation makes the
    measure primarily respond to detail loss, which is useful when a routing
    policy should protect JPEG/blur cases without incorrectly treating a
    contrast or illumination shift as blur.
    """
    if blur_radius <= 0.0:
        raise ValueError("blur_radius must be positive")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    grayscale = image.convert("L")
    raw = np.asarray(grayscale, dtype=np.float32) / 255.0
    blurred = np.asarray(
        grayscale.filter(ImageFilter.GaussianBlur(radius=float(blur_radius))),
        dtype=np.float32,
    ) / 255.0
    return float(np.std(raw - blurred) / max(float(np.std(raw)), float(eps)))


def threshold_calibrated_logits(
    seg_logits: torch.Tensor,
    thresholds: torch.Tensor,
) -> torch.Tensor:
    """Encode per-image OD/OC mask thresholds as zero-threshold logits.

    A threshold ``t`` on an original segmentation probability is exactly
    equivalent to thresholding ``logit - logit(t)`` at 0.5.  This lets a
    frozen, label-free output artifact carry an auditable per-image threshold
    policy without changing the downstream artifact schema or relying on an
    evaluator-side hidden rule.
    """
    if seg_logits.ndim != 4 or seg_logits.shape[1] != 2:
        raise ValueError("seg_logits must have shape [N, 2, H, W]")
    if thresholds.ndim != 2 or thresholds.shape != seg_logits.shape[:2]:
        raise ValueError("thresholds must have shape [N, 2] matching seg_logits")
    values = thresholds.to(device=seg_logits.device, dtype=torch.float32)
    if not bool(torch.isfinite(values).all()) or not bool(((values > 0.0) & (values < 1.0)).all()):
        raise ValueError("thresholds must be finite values strictly between zero and one")
    offsets = torch.logit(values).unsqueeze(-1).unsqueeze(-1)
    return seg_logits.float() - offsets
