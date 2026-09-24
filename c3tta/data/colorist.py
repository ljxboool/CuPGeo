"""Source-only statistical RGB color matching for domain generalization.

This is an independent implementation of Colorist (arXiv:2608.18915).  It
uses only the published global per-channel mean/standard-deviation equation;
the authors' official code was not available when this baseline was added.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image


def colorist_config(
    augmentation: Mapping[str, Any] | None,
) -> dict[str, bool | float | str]:
    """Validate and normalize the training-only Colorist configuration.

    The style pool is intentionally fixed to the current source-training
    manifest.  There is no option to point at source-validation or target
    images, which makes accidental target-style leakage harder.
    """

    params = augmentation or {}
    raw = params.get("colorist", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("train.augmentation.colorist must be a mapping")

    enabled = bool(raw.get("enabled", False))
    probability = float(raw.get("probability", 0.30))
    epsilon = float(raw.get("epsilon", 1.0e-8))
    clamp = bool(raw.get("clamp", True))
    style_pool = str(raw.get("style_pool", "self_source_train"))
    statistics_space = str(raw.get("statistics_space", "raw_rgb_0_1"))

    if not 0.0 <= probability <= 1.0:
        raise ValueError("Colorist probability must be in [0, 1]")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("Colorist epsilon must be finite and positive")
    if style_pool != "self_source_train":
        raise ValueError(
            "Colorist style_pool must be 'self_source_train'; external style "
            "manifests are forbidden"
        )
    if statistics_space != "raw_rgb_0_1":
        raise ValueError("Colorist statistics_space must be 'raw_rgb_0_1'")
    if enabled and not clamp:
        raise ValueError(
            "Colorist training images must set clamp=true before PIL conversion"
        )

    # The formal baseline isolates Colorist from other photometric policies.
    # Geometry-preserving horizontal flips remain allowed.
    if enabled:
        effective_probabilities = {
            "color_prob": float(params.get("color_prob", 0.4)),
            "channel_gain_prob": float(params.get("channel_gain_prob", 0.0)),
            "gamma_prob": float(params.get("gamma_prob", 0.0)),
            "noise_prob": float(params.get("noise_prob", 0.0)),
        }
        active = [
            name for name, value in effective_probabilities.items() if value > 0.0
        ]
        if active:
            raise ValueError(
                "Colorist must be isolated from other photometric augmentation; "
                f"set these probabilities to zero: {', '.join(active)}"
            )

    return {
        "enabled": enabled,
        "probability": probability,
        "epsilon": epsilon,
        "clamp": clamp,
        "style_pool": style_pool,
        "statistics_space": statistics_space,
    }


def rgb_moments(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Return population RGB mean/std in raw ``[0, 1]`` space."""

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    mean = array.mean(axis=(0, 1), dtype=np.float32)
    std = array.std(axis=(0, 1), dtype=np.float32, ddof=0)
    return mean.astype(np.float32), std.astype(np.float32)


def rgb_moments_from_path(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Decode one source-training image and return its immutable style stats."""

    with Image.open(path) as image:
        return rgb_moments(image)


def colorist_array(
    content: np.ndarray,
    style_mean: np.ndarray,
    style_std: np.ndarray,
    *,
    epsilon: float = 1.0e-8,
    clamp: bool = True,
) -> np.ndarray:
    """Apply global per-channel statistical matching to an RGB float array."""

    array = np.asarray(content, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("Colorist content must have shape [H, W, 3]")
    if not np.isfinite(array).all():
        raise ValueError("Colorist content contains NaN or Inf")
    target_mean = np.asarray(style_mean, dtype=np.float32).reshape(-1)
    target_std = np.asarray(style_std, dtype=np.float32).reshape(-1)
    if target_mean.shape != (3,) or target_std.shape != (3,):
        raise ValueError("Colorist style mean/std must each have shape [3]")
    if not np.isfinite(target_mean).all() or not np.isfinite(target_std).all():
        raise ValueError("Colorist style statistics contain NaN or Inf")
    if (target_std < 0.0).any():
        raise ValueError("Colorist style standard deviation must be non-negative")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("Colorist epsilon must be finite and positive")

    content_mean = array.mean(axis=(0, 1), dtype=np.float32)
    content_std = array.std(axis=(0, 1), dtype=np.float32, ddof=0)
    matched = (
        (array - content_mean[None, None, :])
        / (content_std[None, None, :] + np.float32(epsilon))
        * target_std[None, None, :]
        + target_mean[None, None, :]
    )
    if clamp:
        matched = np.clip(matched, 0.0, 1.0)
    return matched.astype(np.float32, copy=False)


def colorist_image(
    content: Image.Image,
    style_mean: np.ndarray,
    style_std: np.ndarray,
    *,
    epsilon: float = 1.0e-8,
) -> Image.Image:
    """Apply Colorist to a PIL image and return quantized RGB pixels."""

    content_array = np.asarray(content.convert("RGB"), dtype=np.float32) / 255.0
    matched = colorist_array(
        content_array,
        style_mean,
        style_std,
        epsilon=epsilon,
        clamp=True,
    )
    pixels = np.rint(matched * 255.0).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")
