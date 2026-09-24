"""Strict aggregation of private offline-score summaries across source seeds."""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real
from pathlib import Path
import json
import re
import statistics
from typing import Any

from c3tta.engine.offline_evaluation import OFFLINE_SCORE_SCHEMA_VERSION


SEED_AGGREGATE_SCHEMA_VERSION = "c3tta.seed_aggregate.v1"
SEED_AGGREGATE_PRIVACY_NOTICE = (
    "PRIVATE AGGREGATE OUTPUT: derived from private evaluation summaries. "
    "Keep under ignored runs/ storage and publish only appropriately aggregated metrics."
)
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_BASELINE_ALIASES = ("source_only", "source-only")
_PROTOCOL_FIELDS = (
    "mode",
    "reset",
    "trainable_scope",
    "adapt_batch_size",
    "steps_per_batch",
    "tta_hyperparameters",
    "prediction_timing",
)


def load_score_summary(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read offline score summary {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Offline score summary must contain a JSON object: {path}")
    return value


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _finite_metric(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be a finite number")
    return result


def _dataset_identity(summary: Mapping[str, Any], seed: int) -> dict[str, Any]:
    manifest = _require_mapping(
        summary.get("evaluation_manifest"),
        f"seed {seed} evaluation_manifest",
    )
    filename = manifest.get("filename")
    checksum = manifest.get("sha256")
    num_samples = manifest.get("num_samples")
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"seed {seed} evaluation_manifest.filename must be non-empty")
    if not isinstance(checksum, str) or not _SHA256_PATTERN.fullmatch(checksum):
        raise ValueError(f"seed {seed} evaluation_manifest.sha256 must be a SHA256 digest")
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError(f"seed {seed} evaluation_manifest.num_samples must be positive")
    return {
        "filename": filename,
        "sha256": checksum.lower(),
        "num_samples": num_samples,
    }


def _resolve_baseline(methods: set[str], requested: str | None) -> str:
    if requested is not None:
        if requested not in methods:
            raise ValueError(
                f"Baseline method {requested!r} is absent; available methods: {sorted(methods)}"
            )
        return requested
    matches = [name for name in _BASELINE_ALIASES if name in methods]
    if len(matches) != 1:
        raise ValueError(
            "Could not resolve one source-only baseline. Use --baseline when the method is "
            "not uniquely named source_only or source-only."
        )
    return matches[0]


def _protocol_signature(
    metadata: Mapping[str, Any],
    *,
    seed: int,
    method: str,
) -> dict[str, Any]:
    context = f"seed {seed} method {method!r} artifact metadata"
    missing = [field for field in _PROTOCOL_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"{context} is missing protocol fields: {missing}")
    mode = metadata["mode"]
    reset = metadata["reset"]
    trainable_scope = metadata["trainable_scope"]
    timing = metadata["prediction_timing"]
    if mode not in {"online", "transductive"}:
        raise ValueError(f"{context} has invalid mode {mode!r}")
    if reset not in {"domain", "image"}:
        raise ValueError(f"{context} has invalid reset {reset!r}")
    if trainable_scope not in {"adapters", "adapters-ln"}:
        raise ValueError(
            f"{context} has invalid trainable_scope {trainable_scope!r}"
        )
    if timing not in {"source_only", "post_update", "post_domain_adaptation"}:
        raise ValueError(f"{context} has invalid prediction_timing {timing!r}")
    expected_timing = (
        "source_only"
        if method in _BASELINE_ALIASES
        else "post_update" if mode == "online" else "post_domain_adaptation"
    )
    if timing != expected_timing:
        raise ValueError(
            f"{context} prediction_timing {timing!r} is inconsistent with "
            f"method/mode; expected {expected_timing!r}"
        )
    for field in ("adapt_batch_size", "steps_per_batch"):
        value = metadata[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{context} field {field} must be a positive integer")
    hyperparameters = _require_mapping(
        metadata["tta_hyperparameters"], f"{context} tta_hyperparameters"
    )
    return {
        "mode": mode,
        "reset": reset,
        "trainable_scope": trainable_scope,
        "adapt_batch_size": metadata["adapt_batch_size"],
        "steps_per_batch": metadata["steps_per_batch"],
        "tta_hyperparameters": dict(hyperparameters),
        "prediction_timing": timing,
    }


def _describe(values: list[float]) -> dict[str, float | int]:
    if len(values) < 2:
        raise ValueError("Across-seed sample standard deviation requires at least two seeds")
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "num_seeds": len(values),
    }


def aggregate_seed_summaries(
    summaries_by_seed: Mapping[int, Mapping[str, Any]],
    *,
    baseline_method: str | None = None,
    expected_seeds: int = 3,
) -> dict[str, Any]:
    """Aggregate final metrics over independent source-training seeds.

    This function intentionally consumes aggregate ``summary.json`` files only.
    It never treats repeated predictions of the same images as independent
    patients and never pools or recomputes image-level bootstrap samples.
    """
    if (
        isinstance(expected_seeds, bool)
        or not isinstance(expected_seeds, int)
        or expected_seeds < 2
    ):
        raise ValueError("expected_seeds must be at least two")
    if len(summaries_by_seed) != expected_seeds:
        raise ValueError(
            f"Expected exactly {expected_seeds} seed summaries, got {len(summaries_by_seed)}"
        )
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in summaries_by_seed):
        raise ValueError("Seed identifiers must be integers")

    seeds = sorted(summaries_by_seed)
    reference_dataset: dict[str, Any] | None = None
    reference_methods: set[str] | None = None
    reference_metrics: set[str] | None = None
    definitions_by_method: dict[str, Mapping[str, Any]] = {}
    protocol_by_method: dict[str, dict[str, Any]] = {}
    values: dict[str, dict[str, list[float]]] = {}
    bootstrap_presence: dict[str, bool] = {}
    checkpoint_by_seed: dict[str, str] = {}

    for seed in seeds:
        summary = _require_mapping(summaries_by_seed[seed], f"seed {seed} summary")
        if summary.get("schema_version") != OFFLINE_SCORE_SCHEMA_VERSION:
            raise ValueError(
                f"seed {seed} has unsupported score schema {summary.get('schema_version')!r}; "
                f"expected {OFFLINE_SCORE_SCHEMA_VERSION!r}"
            )
        dataset = _dataset_identity(summary, seed)
        if reference_dataset is None:
            reference_dataset = dataset
        elif dataset != reference_dataset:
            raise ValueError(
                f"seed {seed} evaluation dataset identity/sample count differs: "
                f"{dataset} != {reference_dataset}"
            )

        methods = _require_mapping(summary.get("methods"), f"seed {seed} methods")
        if not methods or any(not isinstance(name, str) or not name for name in methods):
            raise ValueError(f"seed {seed} methods must contain non-empty method names")
        method_names = set(methods)
        if reference_methods is None:
            reference_methods = method_names
        elif method_names != reference_methods:
            raise ValueError(
                f"seed {seed} method set differs: {sorted(method_names)} != "
                f"{sorted(reference_methods)}"
            )

        seed_metric_set: set[str] | None = None
        seed_checkpoint: str | None = None
        for method in sorted(method_names):
            method_result = _require_mapping(
                methods[method], f"seed {seed} method {method!r}"
            )
            artifact_metadata = _require_mapping(
                method_result.get("artifact_metadata"),
                f"seed {seed} method {method!r} artifact_metadata",
            )
            if artifact_metadata.get("seed") != seed:
                raise ValueError(
                    f"seed {seed} method {method!r} artifact seed is "
                    f"{artifact_metadata.get('seed')!r}"
                )
            if artifact_metadata.get("method") != method:
                raise ValueError(
                    f"seed {seed} method key {method!r} does not match artifact method "
                    f"{artifact_metadata.get('method')!r}"
                )
            checkpoint_sha256 = artifact_metadata.get("checkpoint_sha256")
            if not isinstance(checkpoint_sha256, str) or not _SHA256_PATTERN.fullmatch(
                checkpoint_sha256
            ):
                raise ValueError(
                    f"seed {seed} method {method!r} artifact checkpoint_sha256 "
                    "must be a SHA256 digest"
                )
            checkpoint_sha256 = checkpoint_sha256.lower()
            if seed_checkpoint is None:
                seed_checkpoint = checkpoint_sha256
            elif checkpoint_sha256 != seed_checkpoint:
                raise ValueError(
                    f"seed {seed} methods do not share one source checkpoint: "
                    f"{checkpoint_sha256} != {seed_checkpoint}"
                )
            protocol = _protocol_signature(
                artifact_metadata,
                seed=seed,
                method=method,
            )
            if method in protocol_by_method and protocol != protocol_by_method[method]:
                raise ValueError(
                    f"seed {seed} method {method!r} protocol differs across seeds: "
                    f"{protocol} != {protocol_by_method[method]}"
                )
            protocol_by_method.setdefault(method, protocol)

            metrics = _require_mapping(
                method_result.get("metrics"), f"seed {seed} method {method!r} metrics"
            )
            metric_names = set(metrics)
            if not metric_names or any(
                not isinstance(name, str) or not name for name in metric_names
            ):
                raise ValueError(f"seed {seed} method {method!r} has invalid metric names")
            if seed_metric_set is None:
                seed_metric_set = metric_names
            elif metric_names != seed_metric_set:
                raise ValueError(
                    f"seed {seed} metric set differs between methods: "
                    f"{sorted(metric_names)} != {sorted(seed_metric_set)}"
                )
            if reference_metrics is None:
                reference_metrics = metric_names
            elif metric_names != reference_metrics:
                raise ValueError(
                    f"seed {seed} method {method!r} metric set differs: "
                    f"{sorted(metric_names)} != {sorted(reference_metrics)}"
                )
            if "num_samples" not in metrics:
                raise ValueError(f"seed {seed} method {method!r} has no num_samples metric")
            metric_sample_count = _finite_metric(
                metrics["num_samples"],
                f"seed {seed} method {method!r} metric num_samples",
            )
            if metric_sample_count != float(dataset["num_samples"]):
                raise ValueError(
                    f"seed {seed} method {method!r} metric num_samples "
                    f"{metric_sample_count} != manifest count {dataset['num_samples']}"
                )

            definitions = _require_mapping(
                method_result.get("metric_definitions"),
                f"seed {seed} method {method!r} metric_definitions",
            )
            if method in definitions_by_method and definitions != definitions_by_method[method]:
                raise ValueError(
                    f"seed {seed} method {method!r} metric definitions differ across seeds"
                )
            definitions_by_method.setdefault(method, dict(definitions))
            for metric, raw_value in metrics.items():
                value = _finite_metric(
                    raw_value, f"seed {seed} method {method!r} metric {metric!r}"
                )
                values.setdefault(method, {}).setdefault(metric, []).append(value)

        assert seed_checkpoint is not None
        checkpoint_by_seed[str(seed)] = seed_checkpoint
        bootstrap_presence[str(seed)] = summary.get("bootstrap") is not None

    assert reference_dataset is not None
    assert reference_methods is not None
    assert reference_metrics is not None
    baseline = _resolve_baseline(reference_methods, baseline_method)

    method_aggregates = {
        method: {
            "protocol": protocol_by_method[method],
            "metrics": {
                metric: _describe(values[method][metric])
                for metric in sorted(reference_metrics)
            }
        }
        for method in sorted(reference_methods)
    }
    deltas: dict[str, Any] = {}
    for method in sorted(reference_methods - {baseline}):
        deltas[method] = {
            "direction": f"{method}-minus-{baseline}",
            "metrics": {
                metric: _describe(
                    [
                        candidate - source
                        for candidate, source in zip(
                            values[method][metric], values[baseline][metric]
                        )
                    ]
                )
                for metric in sorted(reference_metrics)
            },
        }

    return {
        "schema_version": SEED_AGGREGATE_SCHEMA_VERSION,
        "privacy_notice": SEED_AGGREGATE_PRIVACY_NOTICE,
        "evaluation_manifest": reference_dataset,
        "seed_aggregation": {
            "unit": "independent_source_training_seed",
            "seeds": seeds,
            "num_seeds": len(seeds),
            "expected_seeds": expected_seeds,
            "center": "arithmetic_mean",
            "dispersion": "sample_standard_deviation_ddof_1",
        },
        "method_order": sorted(reference_methods),
        "baseline_method": baseline,
        "source_checkpoint_sha256_by_seed": checkpoint_by_seed,
        "methods": method_aggregates,
        "deltas_vs_baseline": deltas,
        "uncertainty_separation": {
            "across_seed": (
                "Mean and sample standard deviation use independent source-training seeds "
                "as the unit."
            ),
            "image_level_paired_bootstrap": (
                "Not pooled or recomputed across seeds. Image-level paired bootstrap in each "
                "input summary is a separate within-test-set uncertainty analysis."
            ),
            "input_bootstrap_present_by_seed": bootstrap_presence,
            "bootstrap_aggregated": False,
            "repeated_images_count_as_independent_patients": False,
        },
    }
