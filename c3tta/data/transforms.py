"""Dependency-light paired image/mask transforms."""

from __future__ import annotations

import random
from typing import Any, Mapping
import numpy as np
from PIL import Image, ImageEnhance
import torch
import torch.nn.functional as F


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _resize_tensor(x: torch.Tensor, size: int, mode: str) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(0)
        squeezed = True
    else:
        squeezed = False
    y = F.interpolate(x.float(), size=(size, size), mode=mode, align_corners=False if mode != "nearest" else None)
    return y.squeeze(0) if squeezed else y


def image_to_tensor(image: Image.Image, size: int = 448) -> torch.Tensor:
    image = image.convert("RGB")
    arr = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def mask_to_tensor(mask: Image.Image | None, size: int | None = 448) -> torch.Tensor | None:
    if mask is None:
        return None
    image = mask.convert("L")
    if size is not None:
        image = image.resize((size, size), Image.Resampling.NEAREST)
    arr = np.asarray(image)
    # Both 0/1 and 0/255 binary masks occur in public retinal datasets.
    return torch.from_numpy((arr > 0).astype(np.float32))[None]


def multiclass_mask_to_pair(
    mask: Image.Image,
    size: int | None = 448,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = mask.convert("L")
    if size is not None:
        image = image.resize((size, size), Image.Resampling.NEAREST)
    arr = np.asarray(image)
    # Accept either integer 0/1/2 masks or 0/127/255-style masks.
    if arr.max() > 2:
        labels = np.zeros_like(arr, dtype=np.uint8)
        labels[arr >= 200] = 2
        labels[(arr >= 50) & (arr < 200)] = 1
    else:
        labels = arr.astype(np.uint8)
    od = torch.from_numpy((labels > 0).astype(np.float32))[None]
    oc = torch.from_numpy((labels == 2).astype(np.float32))[None]
    return od, oc


def multiclass_mask_to_images(mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    arr = np.asarray(mask.convert("L"))
    if arr.max() > 2:
        labels = np.zeros_like(arr, dtype=np.uint8)
        labels[arr >= 200] = 2
        labels[(arr >= 50) & (arr < 200)] = 1
    else:
        labels = arr.astype(np.uint8)
    od = Image.fromarray(((labels > 0) * 255).astype(np.uint8), mode="L")
    oc = Image.fromarray(((labels == 2) * 255).astype(np.uint8), mode="L")
    return od, oc


def scale_canvas_pair(
    image: Image.Image,
    od_mask: Image.Image | None,
    oc_mask: Image.Image | None,
    *,
    scale: float,
    center_x_fraction: float = 0.5,
    center_y_fraction: float = 0.5,
) -> tuple[Image.Image, Image.Image | None, Image.Image | None]:
    """Embed one source crop in an edge-padded canvas at a fixed scale.

    The operation preserves the original canvas dimensions. It is a source-only
    geometry transform for simulating crop/FOV variation, not a target-derived
    preprocessing step. Image padding extends border pixels; mask padding is
    always zero so OD/OC geometry remains auditable.
    """

    if not 0.0 < scale <= 1.0:
        raise ValueError(f"scale must be in (0, 1], got {scale}")
    if not 0.0 <= center_x_fraction <= 1.0 or not 0.0 <= center_y_fraction <= 1.0:
        raise ValueError("canvas center fractions must be in [0, 1]")
    source = image.convert("RGB")
    width, height = source.size
    if width <= 0 or height <= 0:
        raise ValueError("cannot scale an empty image")
    for name, mask in (("OD", od_mask), ("OC", oc_mask)):
        if mask is not None and mask.size != source.size:
            raise ValueError(f"{name} mask size does not match source image")

    scaled_width = min(width, max(1, int(round(width * scale))))
    scaled_height = min(height, max(1, int(round(height * scale))))
    resized = source.resize((scaled_width, scaled_height), Image.Resampling.BILINEAR)
    max_left = width - scaled_width
    max_top = height - scaled_height
    left = min(
        max(int(round(center_x_fraction * width - scaled_width / 2.0)), 0),
        max_left,
    )
    top = min(
        max(int(round(center_y_fraction * height - scaled_height / 2.0)), 0),
        max_top,
    )
    right = max_left - left
    bottom = max_top - top
    array = np.asarray(resized, dtype=np.uint8)
    padded = np.pad(array, ((top, bottom), (left, right), (0, 0)), mode="edge")
    transformed_image = Image.fromarray(padded, mode="RGB")

    def transform_mask(mask: Image.Image | None) -> Image.Image | None:
        if mask is None:
            return None
        resized_mask = mask.convert("L").resize(
            (scaled_width, scaled_height), Image.Resampling.NEAREST
        )
        mask_array = np.asarray(resized_mask, dtype=np.uint8)
        padded_mask = np.pad(
            mask_array, ((top, bottom), (left, right)), mode="constant", constant_values=0
        )
        return Image.fromarray(padded_mask, mode="L")

    return transformed_image, transform_mask(od_mask), transform_mask(oc_mask)


def paired_transform(
    image: Image.Image,
    od_mask: Image.Image | None = None,
    oc_mask: Image.Image | None = None,
    size: int = 448,
    train: bool = False,
    augmentation: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Apply geometry-preserving source-training augmentation.

    ``augmentation`` is deliberately optional so existing manifests and
    configurations keep their historical behavior.  The image transforms are
    photometric (or a horizontal flip), which preserves OD/OC mask alignment.
    """
    params = augmentation or {}
    flip_prob = float(params.get("flip_prob", 0.5))
    color_prob = float(params.get("color_prob", 0.4))
    brightness_min = float(params.get("brightness_min", 0.85))
    brightness_max = float(params.get("brightness_max", 1.15))
    contrast_min = float(params.get("contrast_min", 0.85))
    contrast_max = float(params.get("contrast_max", 1.15))
    channel_gain_prob = float(params.get("channel_gain_prob", 0.0))
    channel_gain_min = float(params.get("channel_gain_min", 1.0))
    channel_gain_max = float(params.get("channel_gain_max", 1.0))
    gamma_prob = float(params.get("gamma_prob", 0.0))
    gamma_min = float(params.get("gamma_min", 1.0))
    gamma_max = float(params.get("gamma_max", 1.0))
    noise_prob = float(params.get("noise_prob", 0.0))
    noise_std = float(params.get("noise_std", 0.0))
    row_noise_std = float(params.get("row_noise_std", 0.0))
    scale_canvas_prob = float(params.get("scale_canvas_prob", 0.0))
    scale_canvas_min = float(params.get("scale_canvas_min", 1.0))
    scale_canvas_max = float(params.get("scale_canvas_max", 1.0))
    scale_canvas_center_jitter = float(params.get("scale_canvas_center_jitter", 0.0))
    scale_canvas_log_uniform = bool(params.get("scale_canvas_log_uniform", True))
    if not 0.0 <= flip_prob <= 1.0:
        raise ValueError(f"flip_prob must be in [0, 1], got {flip_prob}")
    if not 0.0 <= color_prob <= 1.0:
        raise ValueError(f"color_prob must be in [0, 1], got {color_prob}")
    if not (0.0 < brightness_min <= brightness_max):
        raise ValueError("brightness range must satisfy 0 < min <= max")
    if not (0.0 < contrast_min <= contrast_max):
        raise ValueError("contrast range must satisfy 0 < min <= max")
    if not 0.0 <= channel_gain_prob <= 1.0:
        raise ValueError(f"channel_gain_prob must be in [0, 1], got {channel_gain_prob}")
    if not (0.0 < channel_gain_min <= channel_gain_max):
        raise ValueError("channel gain range must satisfy 0 < min <= max")
    if not 0.0 <= gamma_prob <= 1.0:
        raise ValueError(f"gamma_prob must be in [0, 1], got {gamma_prob}")
    if not (0.0 < gamma_min <= gamma_max):
        raise ValueError("gamma range must satisfy 0 < min <= max")
    if not 0.0 <= noise_prob <= 1.0:
        raise ValueError(f"noise_prob must be in [0, 1], got {noise_prob}")
    if noise_std < 0.0 or row_noise_std < 0.0:
        raise ValueError("noise_std and row_noise_std must be non-negative")
    if not 0.0 <= scale_canvas_prob <= 1.0:
        raise ValueError("scale_canvas_prob must be in [0, 1]")
    if not (0.0 < scale_canvas_min <= scale_canvas_max <= 1.0):
        raise ValueError("scale canvas range must satisfy 0 < min <= max <= 1")
    if not 0.0 <= scale_canvas_center_jitter <= 0.5:
        raise ValueError("scale_canvas_center_jitter must be in [0, 0.5]")

    if train and scale_canvas_prob > 0.0 and random.random() < scale_canvas_prob:
        if scale_canvas_log_uniform:
            log_scale = random.uniform(
                float(np.log(scale_canvas_min)), float(np.log(scale_canvas_max))
            )
            canvas_scale = float(np.exp(log_scale))
        else:
            canvas_scale = random.uniform(scale_canvas_min, scale_canvas_max)
        center_x = 0.5 + random.uniform(
            -scale_canvas_center_jitter, scale_canvas_center_jitter
        )
        center_y = 0.5 + random.uniform(
            -scale_canvas_center_jitter, scale_canvas_center_jitter
        )
        image, od_mask, oc_mask = scale_canvas_pair(
            image,
            od_mask,
            oc_mask,
            scale=canvas_scale,
            center_x_fraction=center_x,
            center_y_fraction=center_y,
        )

    if train and random.random() < flip_prob:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if od_mask is not None:
            od_mask = od_mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if oc_mask is not None:
            oc_mask = oc_mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if train and random.random() < color_prob:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(brightness_min, brightness_max))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(contrast_min, contrast_max))
    apply_channel_gain = train and channel_gain_prob > 0.0 and random.random() < channel_gain_prob
    apply_gamma = train and gamma_prob > 0.0 and random.random() < gamma_prob
    if apply_channel_gain or apply_gamma:
        # Apply camera response changes in raw RGB space, before normalization.
        # Channel gains approximate white-balance/sensor spectral variation;
        # gamma approximates a nonlinear camera/display response.  Both are
        # optional and default to identity to preserve historical configs.
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        if apply_channel_gain:
            gains = np.asarray(
                [random.uniform(channel_gain_min, channel_gain_max) for _ in range(3)],
                dtype=np.float32,
            )
            array *= gains[None, None, :]
        if apply_gamma:
            gamma = random.uniform(gamma_min, gamma_max)
            array = np.power(np.clip(array, 0.0, 1.0), gamma)
        image = Image.fromarray(
            np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB"
        )
    if train and noise_prob > 0.0 and random.random() < noise_prob:
        # Add noise before resizing so its frequency content is transformed in
        # the same order as the camera-noise source-validation proxy.  Values
        # are specified in raw [0, 1] RGB space (e.g. 7 / 255 ~= 0.027).
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        if noise_std > 0.0:
            array += np.random.normal(0.0, noise_std, size=array.shape).astype(np.float32)
        if row_noise_std > 0.0:
            row_noise = np.random.normal(
                0.0, row_noise_std, size=(array.shape[0], 1, 1)
            ).astype(np.float32)
            array += row_noise
        image = Image.fromarray(np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
    return image_to_tensor(image, size), mask_to_tensor(od_mask, size), mask_to_tensor(oc_mask, size)


def photometric_pair(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Create weak/strong views while preserving geometry for segmentation consistency."""
    weak = image
    strong = image + torch.randn_like(image) * 0.015
    strong = strong.clamp(-3.0, 3.0)
    if random.random() < 0.5:
        strong = strong * random.uniform(0.9, 1.1)
    return weak, strong
