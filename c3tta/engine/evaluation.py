"""Shared, label-only evaluation utilities.

The adaptation loader never enters this module.  Keeping evaluation separate
from target adaptation makes the source-free boundary explicit in code and
prevents accidentally using target labels for updates or early stopping.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import torch

from c3tta.engine.common import autocast_context
from c3tta.metrics import (
    classification_metrics,
    dice_score,
    iou_score,
    vcdr_mae,
)


def normalize_segmentation_thresholds(
    thresholds: float | Sequence[float],
) -> tuple[float, float]:
    """Validate one shared or two task-specific hard-mask thresholds.

    OD and OC occupy very different fractions of a fundus image.  Offline
    source-only calibration may therefore use one fixed threshold per task,
    while the default remains the conventional 0.5 for both.  Keeping this
    normalization in the label-only evaluation module prevents an inference
    or adaptation path from ever consulting the thresholds.
    """
    if isinstance(thresholds, (int, float)):
        values = (float(thresholds), float(thresholds))
    else:
        values = tuple(float(value) for value in thresholds)
        if len(values) != 2:
            raise ValueError(
                "segmentation thresholds must be one shared value or exactly two values (OD, OC)"
            )
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ValueError("segmentation thresholds must be finite values in [0, 1]")
    return values


class MetricAccumulator:
    """Accumulate predictions without exposing labels to the adaptation loss."""

    def __init__(self, segmentation_thresholds: float | Sequence[float] = 0.5) -> None:
        self.segmentation_thresholds = normalize_segmentation_thresholds(
            segmentation_thresholds
        )
        self.dice_by_task: list[list[float]] = [[], []]
        self.iou_by_task: list[list[float]] = [[], []]
        self.cls_logits: list[torch.Tensor] = []
        self.cls_targets: list[torch.Tensor] = []
        self.vcdr_preds: list[torch.Tensor] = []
        self.vcdr_targets: list[torch.Tensor] = []
        self.violation_sum = 0.0
        self.seg_count = 0
        self.sample_count = 0

    def update(self, output: Any, batch: dict[str, Any]) -> None:
        n = int(output.seg_logits.shape[0])
        self.sample_count += n
        if "seg_target" in batch:
            # Prediction artifacts store segmentation logits as float16 to
            # limit their size. Promote before CPU sigmoid, which does not
            # support half precision in all supported PyTorch versions.
            probs = torch.sigmoid(output.seg_logits.detach().float())
            seg_target = batch["seg_target"].to(probs.device)
            for task in range(2):
                self.dice_by_task[task].extend(
                    dice_score(
                        probs[:, task],
                        seg_target[:, task],
                        threshold=self.segmentation_thresholds[task],
                    ).tolist()
                )
                self.iou_by_task[task].extend(
                    iou_score(
                        probs[:, task],
                        seg_target[:, task],
                        threshold=self.segmentation_thresholds[task],
                    ).tolist()
                )
            hard_predictions = torch.stack(
                [
                    probs[:, task] >= self.segmentation_thresholds[task]
                    for task in range(2)
                ],
                dim=1,
            )
            # A case is invalid if any predicted OC pixel lies outside OD.
            violations = (hard_predictions[:, 1] > hard_predictions[:, 0]).flatten(1).any(dim=1)
            self.violation_sum += float(violations.float().mean().item()) * n
            self.seg_count += n
            if "vcdr_target" in batch:
                self.vcdr_preds.append(output.soft_vcdr.detach().cpu())
                self.vcdr_targets.append(batch["vcdr_target"].cpu())
        if "cls_target" in batch:
            self.cls_logits.append(output.cls_logits.detach().cpu())
            self.cls_targets.append(batch["cls_target"].cpu())

    def compute(self) -> dict[str, float]:
        result: dict[str, float] = {"num_samples": float(self.sample_count)}
        if any(self.dice_by_task):
            result["od_dice"] = float(sum(self.dice_by_task[0]) / max(len(self.dice_by_task[0]), 1))
            result["oc_dice"] = float(sum(self.dice_by_task[1]) / max(len(self.dice_by_task[1]), 1))
            result["od_iou"] = float(sum(self.iou_by_task[0]) / max(len(self.iou_by_task[0]), 1))
            result["oc_iou"] = float(sum(self.iou_by_task[1]) / max(len(self.iou_by_task[1]), 1))
            result["dice_mean"] = (result["od_dice"] + result["oc_dice"]) / 2.0
            result["topology_violation"] = self.violation_sum / max(self.seg_count, 1)
        if self.cls_logits:
            result.update(classification_metrics(torch.cat(self.cls_logits), torch.cat(self.cls_targets)))
        if self.vcdr_preds:
            result["vcdr_mae"] = vcdr_mae(torch.cat(self.vcdr_preds), torch.cat(self.vcdr_targets))
        return result


def evaluate_loader(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    precision: str = "fp32",
) -> dict[str, float]:
    """Evaluate a model on a manifest that may contain one or more labels."""
    model.eval()
    accumulator = MetricAccumulator()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with autocast_context(device, precision):
                output = model(images)
            accumulator.update(output, batch)
    return accumulator.compute()
