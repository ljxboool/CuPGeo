"""Offline, label-isolated scoring and paired cluster bootstrap utilities.

Prediction artifacts are produced without labels by the strict TTA inference
path.  This module joins an artifact to a labeled evaluation manifest only
after inference is complete.  Per-image outputs therefore remain private
evaluation records and must not be published with restricted datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import math

import numpy as np
from PIL import Image
import torch

from c3tta.data.manifest import ManifestRow, read_manifest
from c3tta.data.transforms import mask_to_tensor, multiclass_mask_to_pair
from c3tta.engine.evaluation import MetricAccumulator, normalize_segmentation_thresholds
from c3tta.engine.predictions import load_prediction_artifact, validate_prediction_artifact
from c3tta.metrics import classification_metrics, dice_score, iou_score


OFFLINE_SCORE_SCHEMA_VERSION = "c3tta.offline_scores.v1"
DEFAULT_BOOTSTRAP_METRICS = (
    "od_dice",
    "oc_dice",
    "dice_mean",
    "od_iou",
    "oc_iou",
    "topology_violation",
    "f1",
    "positive_f1",
    "macro_f1",
    "auroc",
    "auprc",
    "average_precision",
    "sensitivity",
    "specificity",
    "ece",
    "brier",
    "vcdr_mae",
)


@dataclass(frozen=True)
class OfflineScore:
    """Aggregate and private per-image results for one prediction artifact."""

    metrics: dict[str, float]
    records: list[dict[str, Any]]
    image_ids: list[str]
    artifact_metadata: dict[str, Any]
    metric_definitions: dict[str, str]


def _resolve(root: Path, value: str | None, image_id: str, field: str) -> Path:
    if not value:
        raise ValueError(f"Evaluation row {image_id} has no {field} path")
    path = Path(value)
    resolved = path if path.is_absolute() else root / path
    if not resolved.is_file():
        raise FileNotFoundError(f"{field} not found for {image_id}: {resolved}")
    return resolved


def _load_segmentation_target(
    row: ManifestRow,
    root: Path,
    image_size: int | None,
) -> torch.Tensor:
    if row.mask_encoding.lower().startswith("three_class"):
        mask_path = _resolve(root, row.mask or row.od_mask or row.oc_mask, row.image_id, "mask")
        with Image.open(mask_path) as mask:
            od, oc = multiclass_mask_to_pair(mask, image_size)
    else:
        od_path = _resolve(root, row.od_mask, row.image_id, "od_mask")
        oc_path = _resolve(root, row.oc_mask, row.image_id, "oc_mask")
        with Image.open(od_path) as mask:
            od = mask_to_tensor(mask, image_size)
        with Image.open(oc_path) as mask:
            oc = mask_to_tensor(mask, image_size)
        if od is None or oc is None:  # pragma: no cover - guarded by _resolve
            raise ValueError(f"Could not load segmentation masks for {row.image_id}")
    return torch.cat((od, oc), dim=0)


def _task_availability(rows: Sequence[ManifestRow]) -> tuple[bool, bool, bool]:
    seg_flags = [
        bool((row.od_mask and row.oc_mask) or row.mask or row.mask_encoding.lower().startswith("three_class"))
        for row in rows
    ]
    cls_flags = [row.glaucoma is not None for row in rows]
    vcdr_flags = [row.vcdr is not None for row in rows]
    for name, flags in (("segmentation", seg_flags), ("classification", cls_flags), ("vCDR", vcdr_flags)):
        if any(flags) and not all(flags):
            raise ValueError(f"Evaluation manifest has incomplete {name} labels")
    has_seg, has_cls, has_vcdr = all(seg_flags), all(cls_flags), all(vcdr_flags)
    if not (has_seg or has_cls or has_vcdr):
        raise ValueError("Evaluation manifest contains no usable labels")
    if has_vcdr and not has_seg:
        # MetricAccumulator only consumes vCDR when a segmentation target is
        # present. Scoring it independently is nevertheless well-defined.
        has_vcdr = True
    return has_seg, has_cls, has_vcdr


def _validate_ordered_ids(artifact_ids: Sequence[str], rows: Sequence[ManifestRow]) -> list[str]:
    manifest_ids = [row.image_id for row in rows]
    if len(set(manifest_ids)) != len(manifest_ids):
        raise ValueError("Evaluation manifest contains duplicate image_id values")
    artifact_ids = list(artifact_ids)
    if artifact_ids != manifest_ids:
        first = next(
            (index for index, pair in enumerate(zip(artifact_ids, manifest_ids)) if pair[0] != pair[1]),
            min(len(artifact_ids), len(manifest_ids)),
        )
        artifact_id = artifact_ids[first] if first < len(artifact_ids) else "<missing>"
        manifest_id = manifest_ids[first] if first < len(manifest_ids) else "<missing>"
        missing = sorted(set(manifest_ids) - set(artifact_ids))[:5]
        extra = sorted(set(artifact_ids) - set(manifest_ids))[:5]
        raise ValueError(
            "Prediction artifact and evaluation manifest have different ordered image_id values: "
            f"first mismatch at index {first}: artifact={artifact_id!r}, manifest={manifest_id!r}; "
            f"missing={missing}, extra={extra}"
        )
    return manifest_ids


def _vertical_extent(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return inclusive vertical extent and a non-empty flag for [N,H,W] masks."""
    occupied = mask.bool().any(dim=-1)
    height = occupied.shape[-1]
    rows = torch.arange(height, device=mask.device).view(1, height)
    first = torch.where(occupied, rows, height).amin(dim=-1)
    last = torch.where(occupied, rows, -1).amax(dim=-1)
    valid = occupied.any(dim=-1)
    extent = torch.where(valid, last - first + 1, 0).float()
    return extent, valid


def _mask_vertical_cdr(masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cup/disc vertical-extent ratio from [N,2,H,W] binary masks."""
    disc_extent, disc_valid = _vertical_extent(masks[:, 0])
    cup_extent, _ = _vertical_extent(masks[:, 1])
    ratio = cup_extent / disc_extent.clamp_min(1.0)
    return ratio, disc_valid


def _finite_mean(values: Sequence[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def score_prediction_artifact(
    artifact_or_path: str | Path | Mapping[str, Any],
    eval_manifest: str | Path,
    data_root: str | Path = ".",
    batch_size: int = 16,
    segmentation_thresholds: float | Sequence[float] = 0.5,
) -> OfflineScore:
    """Score one strict prediction artifact against an exactly ordered manifest.

    The artifact must contain no labels.  The function refuses set-equal but
    differently ordered IDs because strict online predictions are sequence
    dependent and positional reassignment would silently invalidate results.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    thresholds = normalize_segmentation_thresholds(segmentation_thresholds)
    artifact = (
        load_prediction_artifact(artifact_or_path)
        if isinstance(artifact_or_path, (str, Path))
        else validate_prediction_artifact(dict(artifact_or_path))
    )
    try:
        rows = read_manifest(eval_manifest)
    except KeyError as exc:
        if exc.args == ("patient_id",):
            raise ValueError(
                "Evaluation manifest is missing the required patient_id column"
            ) from exc
        raise
    if not rows:
        raise ValueError("Evaluation manifest is empty")
    image_ids = _validate_ordered_ids(artifact["image_ids"], rows)
    has_seg, has_cls, has_explicit_vcdr = _task_availability(rows)

    predictions = artifact["predictions"]
    # Prediction artifacts compact segmentation logits to FP16. Some supported
    # CPU-only PyTorch builds do not implement sigmoid for Half tensors, so
    # scoring is deliberately performed in FP32 after artifact quantization.
    seg_logits = predictions["seg_logits"].detach().float().cpu()
    cls_logits = predictions["cls_logits"].detach().float().cpu()
    soft_vcdr = predictions["soft_vcdr"].detach().float().cpu()
    if seg_logits.shape[-2] != seg_logits.shape[-1]:
        raise ValueError(
            "Offline mask scoring currently requires square prediction logits; "
            f"got spatial shape {tuple(seg_logits.shape[-2:])}"
        )
    image_size = int(seg_logits.shape[-1])
    root = Path(data_root)
    accumulator = MetricAccumulator(segmentation_thresholds=thresholds)
    records: list[dict[str, Any]] = []
    definitions: dict[str, str] = {
        "segmentation": (
            "Per-image hard masks from sigmoid(logit) using fixed source-selected thresholds "
            f"OD={thresholds[0]:.6f}, OC={thresholds[1]:.6f}; image metrics are macro-averaged."
        ),
        "classification": "Glaucoma probability is sigmoid(classification logit); decision threshold is 0.5.",
    }

    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        batch_rows = rows[start:stop]
        batch: dict[str, Any] = {"image_id": image_ids[start:stop]}
        native_vcdr_target: torch.Tensor | None = None
        native_vcdr_valid: torch.Tensor | None = None
        if has_seg:
            batch["seg_target"] = torch.stack(
                [_load_segmentation_target(row, root, image_size) for row in batch_rows]
            )
            # Dice follows each artifact's image grid, but the reference vCDR
            # must be independent of an artifact's output resolution.  Keep a
            # native-mask copy for that target so 640- and 768-pixel predictions
            # can be compared by a strict paired bootstrap. Native image sizes
            # vary within real datasets, so calculate each ratio before batching.
            native_vcdr = [
                _mask_vertical_cdr(_load_segmentation_target(row, root, None).unsqueeze(0) >= 0.5)
                for row in batch_rows
            ]
            native_vcdr_target = torch.cat([ratio for ratio, _ in native_vcdr])
            native_vcdr_valid = torch.cat([valid for _, valid in native_vcdr])
        if has_cls:
            batch["cls_target"] = torch.tensor(
                [[float(row.glaucoma)] for row in batch_rows], dtype=torch.float32
            )
        if has_explicit_vcdr and has_seg:
            batch["vcdr_target"] = torch.tensor(
                [[float(row.vcdr)] for row in batch_rows], dtype=torch.float32
            )

        output = SimpleNamespace(
            seg_logits=seg_logits[start:stop],
            cls_logits=cls_logits[start:stop],
            soft_vcdr=soft_vcdr[start:stop],
        )
        accumulator.update(output, batch)
        batch_records = [
            {"image_id": row.image_id, "patient_id": row.patient_id}
            for row in batch_rows
        ]

        if has_seg:
            probs = torch.sigmoid(output.seg_logits.detach())
            targets = batch["seg_target"]
            od_dice = dice_score(probs[:, 0], targets[:, 0], threshold=thresholds[0])
            oc_dice = dice_score(probs[:, 1], targets[:, 1], threshold=thresholds[1])
            od_iou = iou_score(probs[:, 0], targets[:, 0], threshold=thresholds[0])
            oc_iou = iou_score(probs[:, 1], targets[:, 1], threshold=thresholds[1])
            hard_predictions = torch.stack(
                [
                    probs[:, task] >= thresholds[task]
                    for task in range(2)
                ],
                dim=1,
            )
            violations = (hard_predictions[:, 1] > hard_predictions[:, 0]).flatten(1).any(dim=1)
            for offset, record in enumerate(batch_records):
                record.update(
                    od_dice=float(od_dice[offset]),
                    oc_dice=float(oc_dice[offset]),
                    dice_mean=float((od_dice[offset] + oc_dice[offset]) / 2.0),
                    od_iou=float(od_iou[offset]),
                    oc_iou=float(oc_iou[offset]),
                    topology_violation=float(violations[offset]),
                )

        if has_cls:
            probabilities = torch.sigmoid(output.cls_logits).flatten()
            for offset, record in enumerate(batch_records):
                target = int(batch_rows[offset].glaucoma)
                probability = float(probabilities[offset])
                record.update(
                    cls_logit=float(output.cls_logits[offset].flatten()[0]),
                    glaucoma_probability=probability,
                    glaucoma_prediction=int(probability >= 0.5),
                    glaucoma_target=target,
                    brier_error=float((probability - target) ** 2),
                )

        if has_explicit_vcdr:
            definitions["vcdr"] = (
                "Absolute error between artifact soft_vcdr and the explicit manifest vCDR target."
            )
            for offset, record in enumerate(batch_records):
                prediction = float(output.soft_vcdr[offset].flatten()[0])
                target = float(batch_rows[offset].vcdr)
                record.update(
                    vcdr_prediction=prediction,
                    vcdr_target=target,
                    vcdr_abs_error=abs(prediction - target),
                    vcdr_valid=True,
                )
        elif has_seg:
            definitions["vcdr"] = (
                "Mask-derived vertical CDR: inclusive vertical cup extent divided by inclusive "
                "vertical disc extent on the native GT mask; prediction masks use sigmoid(logit) "
                f">= fixed thresholds OD={thresholds[0]:.6f}, OC={thresholds[1]:.6f} on the artifact "
                "grid. Empty predicted discs are undefined and reported."
            )
            predicted_vcdr, predicted_valid = _mask_vertical_cdr(hard_predictions)
            if native_vcdr_target is None or native_vcdr_valid is None:  # pragma: no cover
                raise RuntimeError("Missing native segmentation targets for mask-derived vCDR")
            target_vcdr, target_valid = native_vcdr_target, native_vcdr_valid
            if not bool(target_valid.all()):
                invalid_ids = [
                    batch_rows[index].image_id
                    for index, valid in enumerate(target_valid.tolist())
                    if not valid
                ]
                raise ValueError(f"Ground-truth OD mask is empty for image_id values {invalid_ids}")
            for offset, record in enumerate(batch_records):
                valid = bool(predicted_valid[offset])
                prediction = float(predicted_vcdr[offset]) if valid else None
                target = float(target_vcdr[offset])
                record.update(
                    vcdr_prediction=prediction,
                    vcdr_target=target,
                    vcdr_abs_error=abs(prediction - target) if prediction is not None else None,
                    vcdr_valid=valid,
                )
        records.extend(batch_records)

    metrics = accumulator.compute()
    if has_explicit_vcdr and not has_seg:
        errors = [record["vcdr_abs_error"] for record in records]
        metrics["vcdr_mae"] = float(np.mean(errors))
    elif not has_explicit_vcdr and has_seg:
        errors = [record["vcdr_abs_error"] for record in records]
        valid_mean = _finite_mean(errors)
        if valid_mean is not None:
            metrics["vcdr_mae"] = valid_mean
        valid_count = sum(record["vcdr_valid"] for record in records)
        metrics["vcdr_valid_samples"] = float(valid_count)
        metrics["vcdr_invalid_predictions"] = float(len(records) - valid_count)

    return OfflineScore(
        metrics=metrics,
        records=records,
        image_ids=image_ids,
        artifact_metadata=dict(artifact["metadata"]),
        metric_definitions=definitions,
    )


def aggregate_per_image_records(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Recompute aggregate metrics from private records, including resamples."""
    if not records:
        raise ValueError("Cannot aggregate an empty record sequence")
    result: dict[str, float] = {"num_samples": float(len(records))}
    mean_fields = (
        "od_dice",
        "oc_dice",
        "dice_mean",
        "od_iou",
        "oc_iou",
        "topology_violation",
    )
    for field in mean_fields:
        values = [record.get(field) for record in records]
        mean = _finite_mean(values)
        if mean is not None:
            result[field] = mean

    if all(record.get("cls_logit") is not None and record.get("glaucoma_target") is not None for record in records):
        logits = torch.tensor([[float(record["cls_logit"])] for record in records])
        targets = torch.tensor([[float(record["glaucoma_target"])] for record in records])
        result.update(classification_metrics(logits, targets))

    vcdr_errors = [record.get("vcdr_abs_error") for record in records]
    if any(value is not None for value in vcdr_errors):
        mean = _finite_mean(vcdr_errors)
        if mean is not None:
            result["vcdr_mae"] = mean
        valid = sum(value is not None and math.isfinite(float(value)) for value in vcdr_errors)
        result["vcdr_valid_samples"] = float(valid)
        result["vcdr_invalid_predictions"] = float(len(records) - valid)
    return result


def _percentile_interval(
    values: np.ndarray,
    confidence: float,
) -> tuple[float | None, float | None, int]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None, 0
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(finite, (alpha, 1.0 - alpha))
    return float(lower), float(upper), int(finite.size)


def _patient_cluster_indices(
    records: Sequence[Mapping[str, Any]],
    method: str,
) -> tuple[list[np.ndarray], list[str]]:
    """Return first-occurrence ordered patient clusters after strict validation."""
    indices_by_patient: dict[str, list[int]] = {}
    patient_ids: list[str] = []
    for index, record in enumerate(records):
        patient_id = record.get("patient_id")
        if not isinstance(patient_id, str) or not patient_id.strip():
            image_id = record.get("image_id", f"index {index}")
            raise ValueError(
                f"Patient bootstrap requires a non-empty patient_id for method "
                f"{method!r}, image_id {image_id!r}"
            )
        patient_ids.append(patient_id)
        indices_by_patient.setdefault(patient_id, []).append(index)
    clusters = [
        np.asarray(indices, dtype=np.int64) for indices in indices_by_patient.values()
    ]
    return clusters, patient_ids


def paired_image_bootstrap(
    scores: Mapping[str, OfflineScore | Sequence[Mapping[str, Any]]],
    num_resamples: int = 10_000,
    seed: int = 2026,
    confidence: float = 0.95,
    metrics: Sequence[str] | None = None,
    unit: str = "image",
) -> dict[str, Any]:
    """Compute deterministic image- or patient-cluster percentile intervals.

    One common array of image indices or sampled patient clusters is used for
    every method. This preserves pairing for method differences. Patient mode
    samples unique patients with replacement and includes every image belonging
    to each sampled patient.
    """
    if not scores:
        raise ValueError("At least one method is required for bootstrap")
    if num_resamples <= 0:
        raise ValueError("num_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if unit not in {"image", "patient"}:
        raise ValueError("bootstrap unit must be 'image' or 'patient'")

    records_by_method: dict[str, list[Mapping[str, Any]]] = {}
    for method, value in scores.items():
        if not method:
            raise ValueError("Method names must be non-empty")
        records = value.records if isinstance(value, OfflineScore) else list(value)
        if not records:
            raise ValueError(f"Method {method!r} has no per-image records")
        records_by_method[method] = list(records)

    method_names = list(records_by_method)
    reference_records = records_by_method[method_names[0]]
    reference_ids = [str(record["image_id"]) for record in reference_records]
    patient_clusters: list[np.ndarray] | None = None
    reference_patient_ids: list[str] | None = None
    if unit == "patient":
        patient_clusters, reference_patient_ids = _patient_cluster_indices(
            reference_records, method_names[0]
        )
    for method in method_names[1:]:
        method_records = records_by_method[method]
        ids = [str(record["image_id"]) for record in method_records]
        if ids != reference_ids:
            raise ValueError(f"Method {method!r} does not have the same ordered image_id values")
        if unit == "patient":
            _, patient_ids = _patient_cluster_indices(method_records, method)
            if patient_ids != reference_patient_ids:
                first = next(
                    index
                    for index, pair in enumerate(zip(reference_patient_ids, patient_ids))
                    if pair[0] != pair[1]
                )
                raise ValueError(
                    f"Method {method!r} has a different patient_id at index {first}: "
                    f"{patient_ids[first]!r} != {reference_patient_ids[first]!r}"
                )
        for index, (reference, candidate) in enumerate(
            zip(reference_records, method_records)
        ):
            for target_field in ("glaucoma_target", "vcdr_target"):
                if reference.get(target_field) != candidate.get(target_field):
                    raise ValueError(
                        f"Method {method!r} has a different {target_field} at index {index}"
                    )

    point_estimates = {
        method: aggregate_per_image_records(records)
        for method, records in records_by_method.items()
    }
    requested_metrics = tuple(metrics) if metrics is not None else DEFAULT_BOOTSTRAP_METRICS
    if not requested_metrics:
        raise ValueError("At least one bootstrap metric is required")
    unknown = [
        metric
        for metric in requested_metrics
        if not any(metric in estimates for estimates in point_estimates.values())
    ]
    if unknown:
        raise ValueError(f"Requested bootstrap metrics are unavailable: {unknown}")

    rng = np.random.default_rng(seed)
    if unit == "image":
        sample_units = rng.integers(
            0,
            len(reference_ids),
            size=(num_resamples, len(reference_ids)),
            endpoint=False,
        )
        num_clusters = None
        cluster_sizes = None
    else:
        assert patient_clusters is not None
        num_clusters = len(patient_clusters)
        sample_units = rng.integers(
            0,
            num_clusters,
            size=(num_resamples, num_clusters),
            endpoint=False,
        )
        cluster_sizes = [int(indices.size) for indices in patient_clusters]
    distributions: dict[str, dict[str, np.ndarray]] = {
        method: {
            metric: np.full(num_resamples, np.nan, dtype=np.float64)
            for metric in requested_metrics
            if metric in point_estimates[method]
        }
        for method in method_names
    }
    for bootstrap_index, sampled_units in enumerate(sample_units):
        if unit == "image":
            indices = sampled_units
        else:
            assert patient_clusters is not None
            indices = np.concatenate(
                [patient_clusters[int(cluster_index)] for cluster_index in sampled_units]
            )
        for method in method_names:
            source = records_by_method[method]
            sampled = [source[int(index)] for index in indices]
            estimates = aggregate_per_image_records(sampled)
            for metric, values in distributions[method].items():
                value = estimates.get(metric)
                if value is not None and math.isfinite(float(value)):
                    values[bootstrap_index] = float(value)

    method_results: dict[str, dict[str, Any]] = {}
    for method in method_names:
        method_results[method] = {}
        for metric, distribution in distributions[method].items():
            lower, upper, valid = _percentile_interval(distribution, confidence)
            method_results[method][metric] = {
                "estimate": float(point_estimates[method][metric]),
                "ci_lower": lower,
                "ci_upper": upper,
                "valid_resamples": valid,
            }

    comparisons: dict[str, dict[str, Any]] = {}
    for left, right in combinations(method_names, 2):
        comparison_name = f"{right}-minus-{left}"
        comparisons[comparison_name] = {}
        common_metrics = [
            metric
            for metric in requested_metrics
            if metric in distributions[left] and metric in distributions[right]
        ]
        for metric in common_metrics:
            differences = distributions[right][metric] - distributions[left][metric]
            lower, upper, valid = _percentile_interval(differences, confidence)
            finite = differences[np.isfinite(differences)]
            tail_probability = None
            probability_positive = None
            if finite.size:
                probability_positive = float(np.mean(finite > 0.0))
                tail_probability = float(
                    min(1.0, 2.0 * min(np.mean(finite <= 0.0), np.mean(finite >= 0.0)))
                )
            comparisons[comparison_name][metric] = {
                "estimate_difference": float(
                    point_estimates[right][metric] - point_estimates[left][metric]
                ),
                "ci_lower": lower,
                "ci_upper": upper,
                "valid_resamples": valid,
                "probability_difference_positive": probability_positive,
                "two_sided_bootstrap_tail_probability": tail_probability,
            }

    result = {
        "unit": unit,
        "paired": True,
        "method_order": method_names,
        "num_images": len(reference_ids),
        "num_resamples": num_resamples,
        "seed": seed,
        "confidence": confidence,
        "interval": "percentile",
        "methods": method_results,
        "paired_differences": comparisons,
    }
    if unit == "patient":
        assert num_clusters is not None
        assert cluster_sizes is not None
        result.update(
            num_clusters=num_clusters,
            cluster_size_min_images=min(cluster_sizes),
            cluster_size_max_images=max(cluster_sizes),
        )
    return result


def combine_private_records(scores: Mapping[str, OfflineScore]) -> list[dict[str, Any]]:
    """Combine exact-order method records into one private paired JSONL table."""
    if not scores:
        return []
    method_names = list(scores)
    reference = scores[method_names[0]]
    target_fields = ("glaucoma_target", "vcdr_target")
    combined: list[dict[str, Any]] = []
    for index, image_id in enumerate(reference.image_ids):
        row: dict[str, Any] = {
            "image_id": image_id,
            "patient_id": reference.records[index].get("patient_id"),
            "targets": {},
            "methods": {},
        }
        reference_record = reference.records[index]
        row["targets"] = {
            field: reference_record[field]
            for field in target_fields
            if field in reference_record
        }
        for method in method_names:
            score = scores[method]
            if score.image_ids != reference.image_ids:
                raise ValueError(f"Method {method!r} does not have matching ordered image IDs")
            record = score.records[index]
            if record.get("patient_id") != row["patient_id"]:
                raise ValueError(f"Method {method!r} has inconsistent patient_id at {image_id}")
            for field in target_fields:
                if field in row["targets"] and record.get(field) != row["targets"][field]:
                    raise ValueError(f"Method {method!r} has inconsistent target {field} at {image_id}")
            row["methods"][method] = {
                key: value
                for key, value in record.items()
                if key not in {"image_id", "patient_id"} and key not in target_fields
            }
        combined.append(row)
    return combined
