from __future__ import annotations

import math
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix


def dice_score(prob: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> np.ndarray:
    pred = (prob >= threshold).float()
    dims = (1, 2)
    inter = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    return ((2 * inter + eps) / (denom + eps)).detach().cpu().numpy()


def iou_score(prob: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> np.ndarray:
    pred = (prob >= threshold).float()
    inter = (pred * target).sum(dim=(1, 2))
    union = (pred + target - pred * target).sum(dim=(1, 2))
    return ((inter + eps) / (union + eps)).detach().cpu().numpy()


def topology_violation(prob: torch.Tensor, threshold: float = 0.5) -> float:
    pred = (prob >= threshold).float()
    # A case is invalid if any predicted OC pixel lies outside predicted OD.
    violation = (pred[:, 1] > pred[:, 0]).flatten(1).any(dim=1).float()
    return float(violation.mean().item())


def vcdr_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_cpu = pred.detach().float().flatten().cpu()
    target_cpu = target.detach().float().flatten().cpu()
    return float((pred_cpu - target_cpu).abs().mean().item())


def classification_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    y = target.detach().cpu().numpy().reshape(-1).astype(int)
    # NumPy does not support PyTorch BF16 tensors; metrics are accumulated in
    # FP32 without changing the BF16 inference path.
    p = torch.sigmoid(logits).detach().float().cpu().numpy().reshape(-1)
    pred = (p >= 0.5).astype(int)
    positive_f1 = float(f1_score(y, pred, zero_division=0))
    result: dict[str, float] = {
        # Keep ``f1`` as a compatibility alias for historical run files.
        "f1": positive_f1,
        "positive_f1": positive_f1,
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }
    if len(np.unique(y)) > 1:
        result["auroc"] = float(roc_auc_score(y, p))
        average_precision = float(average_precision_score(y, p))
        # ``auprc`` remains as a compatibility alias; sklearn computes AP.
        result["average_precision"] = average_precision
        result["auprc"] = average_precision
    else:
        result.update(
            auroc=float("nan"),
            average_precision=float("nan"),
            auprc=float("nan"),
        )
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    result["sensitivity"] = float(tp / max(tp + fn, 1))
    result["specificity"] = float(tn / max(tn + fp, 1))
    # 10-bin ECE, reported as a diagnostic rather than a training objective.
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    result["ece"] = float(ece)
    result["brier"] = float(np.mean((p - y) ** 2))
    return result
