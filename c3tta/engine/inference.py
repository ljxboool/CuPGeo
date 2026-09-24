"""Label-free prediction-time augmentations shared by inference entry points."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from c3tta.models.multitask import ModelOutput, model_output_from_logits


def _normalise_image_sizes(
    image_sizes: Sequence[int] | None,
    original_height: int,
    original_width: int,
) -> tuple[int, ...]:
    if original_height != original_width:
        raise ValueError("prediction-time multi-scale inference currently requires square images")
    if image_sizes is None:
        return (original_height,)
    result: list[int] = []
    for raw_size in image_sizes:
        size = int(raw_size)
        if size <= 0:
            raise ValueError("prediction-time image sizes must be positive")
        if size not in result:
            result.append(size)
    if not result:
        raise ValueError("prediction-time image_sizes must not be empty")
    return tuple(result)


def predict_with_tta(
    model: nn.Module,
    images: torch.Tensor,
    *,
    image_sizes: Sequence[int] | None = None,
    horizontal_flip: bool = False,
) -> ModelOutput:
    """Average label-free multi-scale and horizontal-flip predictions in logit space.

    Every segmentation logit is resized back to the caller's image grid before
    averaging.  Classification logits are averaged over the same views.  This
    function deliberately has no labels, thresholds, or learned calibration
    parameters, so it is safe for target-adaptation manifests.
    """
    if images.ndim != 4:
        raise ValueError(f"expected [B,C,H,W] images, received {tuple(images.shape)}")
    original_size = tuple(int(value) for value in images.shape[-2:])
    sizes = _normalise_image_sizes(image_sizes, *original_size)
    segmentation_logits: list[torch.Tensor] = []
    classification_logits: list[torch.Tensor] = []

    for size in sizes:
        view = (
            images
            if size == original_size[0]
            else F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
        )
        output = model(view)
        segmentation_logits.append(
            F.interpolate(
                output.seg_logits,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
            if output.seg_logits.shape[-2:] != original_size
            else output.seg_logits
        )
        classification_logits.append(output.cls_logits)

        if horizontal_flip:
            flipped = model(torch.flip(view, dims=(-1,)))
            seg_logits = torch.flip(flipped.seg_logits, dims=(-1,))
            segmentation_logits.append(
                F.interpolate(
                    seg_logits,
                    size=original_size,
                    mode="bilinear",
                    align_corners=False,
                )
                if seg_logits.shape[-2:] != original_size
                else seg_logits
            )
            classification_logits.append(flipped.cls_logits)

    seg_logits = torch.stack(segmentation_logits, dim=0).mean(dim=0)
    cls_logits = torch.stack(classification_logits, dim=0).mean(dim=0)
    return model_output_from_logits(seg_logits, cls_logits)
