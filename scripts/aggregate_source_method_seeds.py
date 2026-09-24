#!/usr/bin/env python3
"""Aggregate distinct source-trained methods over independent seeds.

Unlike the TTA-specific seed aggregator, this utility allows each method to
have its own checkpoint while requiring the same method set, manifest,
precision, and metrics for every seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
from typing import Any, Mapping


SCHEMA_VERSION = "c3tta.source_method_seed_aggregate.v1"
OFFLINE_SCORE_SCHEMA_VERSION = "c3tta.offline_scores.v1"
_SEED_SUMMARY_PATTERN = re.compile(r"^(0|[1-9][0-9]*)=(.+)$")
_LOWER_IS_BETTER = {
    "brier",
    "ece",
    "topology_violation",
    "vcdr_mae",
}


def _parse_seed_summary(value: str) -> tuple[int, Path]:
    match = _SEED_SUMMARY_PATTERN.fullmatch(value)
    if match is None:
        raise argparse.ArgumentTypeError("--seed-summary must use SEED=PATH syntax")
    return int(match.group(1)), Path(match.group(2))


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _describe(values: list[float]) -> dict[str, float | int]:
    if len(values) < 2:
        raise ValueError("At least two seeds are required for sample standard deviation")
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "num_seeds": len(values),
    }


def aggregate_source_method_summaries(
    summaries: Mapping[int, Mapping[str, Any]],
    *,
    baseline: str,
    expected_seeds: int = 3,
    expected_precision: str = "fp32",
) -> dict[str, Any]:
    if len(summaries) != expected_seeds:
        raise ValueError(f"Expected {expected_seeds} seeds, got {len(summaries)}")
    seeds = sorted(summaries)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seed identifiers must be unique")

    reference_manifest: dict[str, Any] | None = None
    reference_thresholds: dict[str, Any] | None = None
    reference_boundary_tolerance: float | None = None
    reference_methods: set[str] | None = None
    reference_metrics: set[str] | None = None
    values: dict[str, dict[str, list[float]]] = {}
    checkpoints: dict[str, dict[str, str]] = {}
    bootstrap_by_seed: dict[str, bool] = {}

    for seed in seeds:
        summary = _mapping(summaries[seed], f"seed {seed} summary")
        if summary.get("schema_version") != OFFLINE_SCORE_SCHEMA_VERSION:
            raise ValueError(f"seed {seed} has an unsupported score schema")
        manifest = dict(_mapping(summary.get("evaluation_manifest"), "evaluation_manifest"))
        thresholds = dict(_mapping(summary.get("segmentation_thresholds"), "segmentation_thresholds"))
        boundary_tolerance = _finite(
            summary.get("boundary_tolerance_fraction", 0.005),
            "boundary_tolerance_fraction",
        )
        if reference_manifest is None:
            reference_manifest = manifest
            reference_thresholds = thresholds
            reference_boundary_tolerance = boundary_tolerance
        elif (
            manifest != reference_manifest
            or thresholds != reference_thresholds
            or boundary_tolerance != reference_boundary_tolerance
        ):
            raise ValueError(
                f"seed {seed} uses a different manifest, threshold, or boundary protocol"
            )

        methods = _mapping(summary.get("methods"), f"seed {seed} methods")
        method_names = set(methods)
        if baseline not in method_names:
            raise ValueError(f"Baseline {baseline!r} is absent from seed {seed}")
        if reference_methods is None:
            reference_methods = method_names
        elif method_names != reference_methods:
            raise ValueError(f"seed {seed} has a different method set")

        bootstrap_by_seed[str(seed)] = summary.get("bootstrap") is not None
        seed_metric_set: set[str] | None = None
        for method in sorted(method_names):
            result = _mapping(methods[method], f"seed {seed} method {method}")
            metadata = _mapping(result.get("artifact_metadata"), "artifact_metadata")
            if metadata.get("method") != method:
                raise ValueError(f"seed {seed} artifact method mismatch for {method}")
            artifact_seed = metadata.get("source_seed", metadata.get("seed"))
            if artifact_seed != seed:
                raise ValueError(
                    f"seed {seed} method {method} records source seed {artifact_seed!r}"
                )
            if metadata.get("inference_precision") != expected_precision:
                raise ValueError(
                    f"seed {seed} method {method} was not inferred with {expected_precision}"
                )
            checkpoint = metadata.get("checkpoint_sha256")
            if not isinstance(checkpoint, str) or len(checkpoint) != 64:
                raise ValueError(f"seed {seed} method {method} lacks checkpoint SHA256")
            checkpoints.setdefault(method, {})[str(seed)] = checkpoint.lower()

            metrics = _mapping(result.get("metrics"), f"seed {seed} method {method} metrics")
            metric_names = set(metrics)
            if seed_metric_set is None:
                seed_metric_set = metric_names
            elif metric_names != seed_metric_set:
                raise ValueError(f"seed {seed} methods expose different metric sets")
            for metric, value in metrics.items():
                values.setdefault(method, {}).setdefault(metric, []).append(
                    _finite(value, f"seed {seed} method {method} metric {metric}")
                )
        if reference_metrics is None:
            reference_metrics = seed_metric_set
        elif seed_metric_set != reference_metrics:
            raise ValueError(f"seed {seed} exposes a different metric set")

    assert reference_manifest is not None
    assert reference_thresholds is not None
    assert reference_boundary_tolerance is not None
    assert reference_methods is not None
    aggregate_methods = {
        method: {
            "metrics": {
                metric: _describe(metric_values)
                for metric, metric_values in sorted(method_metrics.items())
            },
            "checkpoint_sha256_by_seed": checkpoints[method],
        }
        for method, method_metrics in sorted(values.items())
    }

    deltas: dict[str, Any] = {}
    for method in sorted(reference_methods - {baseline}):
        method_deltas: dict[str, Any] = {}
        for metric in sorted(reference_metrics or set()):
            raw = [
                candidate - control
                for candidate, control in zip(values[method][metric], values[baseline][metric])
            ]
            favorable = [-value for value in raw] if metric in _LOWER_IS_BETTER else raw
            method_deltas[metric] = {
                "raw_candidate_minus_baseline": _describe(raw),
                "favorable_improvement": _describe(favorable),
                "lower_is_better": metric in _LOWER_IS_BETTER,
            }
        deltas[method] = method_deltas

    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_manifest": reference_manifest,
        "segmentation_thresholds": reference_thresholds,
        "boundary_tolerance_fraction": reference_boundary_tolerance,
        "seed_protocol": {
            "seeds": seeds,
            "expected_precision": expected_precision,
            "bootstrap_aggregated": False,
            "input_bootstrap_present_by_seed": bootstrap_by_seed,
            "note": "Image/patient bootstrap remains within each seed; repeated images are not pooled across seeds.",
        },
        "baseline_method": baseline,
        "methods": aggregate_methods,
        "deltas_vs_baseline": deltas,
    }


def _private_path(path: Path, private_root: Path, *, must_exist: bool) -> Path:
    root = private_root.resolve()
    resolved = path.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"Refusing path outside private root {root}: {resolved}")
    if must_exist and not resolved.is_file():
        raise ValueError(f"Input summary does not exist: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-summary", action="append", type=_parse_seed_summary, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument("--expected-precision", default="fp32")
    parser.add_argument("--output", required=True)
    parser.add_argument("--private-root", default="runs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summaries: dict[int, Mapping[str, Any]] = {}
    inputs: list[dict[str, Any]] = []
    for seed, raw_path in args.seed_summary:
        if seed in summaries:
            parser.error(f"duplicate seed: {seed}")
        path = _private_path(raw_path, Path(args.private_root), must_exist=True)
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"could not read seed {seed} summary: {exc}")
        summaries[seed] = _mapping(summary, f"seed {seed} summary")
        inputs.append({"seed": seed, "filename": path.name, "sha256": _sha256(path)})

    try:
        result = aggregate_source_method_summaries(
            summaries,
            baseline=args.baseline,
            expected_seeds=args.expected_seeds,
            expected_precision=args.expected_precision,
        )
        output = _private_path(Path(args.output), Path(args.private_root), must_exist=False)
    except ValueError as exc:
        parser.error(str(exc))
    result["inputs"] = inputs

    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output.parent, 0o700)
    if output.exists() and not args.overwrite:
        parser.error(f"output already exists: {output}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
