#!/usr/bin/env python3
"""Score frozen prediction artifacts without exposing labels to adaptation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

# ``score_predictions.py`` is also launched by source-only sweep subprocesses
# with a separate Python environment.  In that case Python places
# ``scripts/`` (rather than the repository root) on ``sys.path``; make the
# package import independent of the caller's ``PYTHONPATH``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from c3tta.engine.offline_evaluation import (
    OFFLINE_SCORE_SCHEMA_VERSION,
    combine_private_records,
    paired_image_bootstrap,
    score_prediction_artifact,
)


METHOD_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
PRIVACY_NOTICE = (
    "PRIVATE EVALUATION OUTPUT: contains per-image labels and derived scores. "
    "Keep under ignored runs/ storage; do not publish or redistribute with restricted datasets."
)


def _parse_prediction(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--prediction must use METHOD=PATH syntax")
    method, raw_path = value.split("=", 1)
    if not METHOD_PATTERN.fullmatch(method):
        raise argparse.ArgumentTypeError(
            "METHOD may contain only letters, digits, dot, underscore, and hyphen"
        )
    if not raw_path:
        raise argparse.ArgumentTypeError("Prediction artifact path must not be empty")
    return method, Path(raw_path)


def _require_private_output(output_dir: Path, private_root: Path) -> Path:
    output = output_dir.resolve()
    root = private_root.resolve()
    if output == root or root not in output.parents:
        raise ValueError(
            f"Refusing per-image label output outside private root {root}; got {output}. "
            "Choose an output directory below runs/."
        )
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_private_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    os.chmod(path, 0o600)


def _validate_artifact_methods(scores: dict[str, Any]) -> None:
    for requested_method, score in scores.items():
        artifact_method = score.artifact_metadata.get("method")
        if artifact_method != requested_method:
            raise ValueError(
                f"Prediction label {requested_method!r} does not match artifact method "
                f"{artifact_method!r}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline scoring and paired image/patient bootstrap for strict prediction artifacts."
    )
    parser.add_argument(
        "--prediction",
        action="append",
        type=_parse_prediction,
        required=True,
        metavar="METHOD=PATH",
        help="Repeat for each frozen method artifact to compare.",
    )
    parser.add_argument("--eval-csv", required=True)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--private-root",
        default="runs",
        help="Per-image output must be below this ignored private directory (default: runs).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--segmentation-threshold",
        action="append",
        type=float,
        default=None,
        metavar="VALUE",
        help=(
            "Fixed hard-mask threshold selected on source data. Supply once for a shared OD/OC "
            "threshold or twice in OD, OC order (default: 0.5, 0.5)."
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument(
        "--bootstrap-unit",
        choices=("image", "patient"),
        default="image",
        help="Resampling unit; patient mode samples whole patient clusters (default: image).",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help="Restrict bootstrap to a metric; repeat for multiple metrics.",
    )
    args = parser.parse_args()

    if args.segmentation_threshold is None:
        segmentation_thresholds: float | tuple[float, float] = 0.5
    elif len(args.segmentation_threshold) == 1:
        segmentation_thresholds = args.segmentation_threshold[0]
    elif len(args.segmentation_threshold) == 2:
        segmentation_thresholds = tuple(args.segmentation_threshold)
    else:
        parser.error("--segmentation-threshold may be supplied once or twice (OD then OC)")

    methods: dict[str, Path] = {}
    for method, path in args.prediction:
        if method in methods:
            parser.error(f"duplicate prediction method: {method}")
        methods[method] = path
    try:
        output_dir = _require_private_output(Path(args.output_dir), Path(args.private_root))
    except ValueError as exc:
        parser.error(str(exc))
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output_dir, 0o700)

    scores = {
        method: score_prediction_artifact(
            path,
            args.eval_csv,
            data_root=args.data_root,
            batch_size=args.batch_size,
            segmentation_thresholds=segmentation_thresholds,
        )
        for method, path in methods.items()
    }
    _validate_artifact_methods(scores)
    bootstrap = None
    if args.bootstrap_samples > 0:
        bootstrap = paired_image_bootstrap(
            scores,
            num_resamples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
            confidence=args.confidence,
            metrics=args.metric,
            unit=args.bootstrap_unit,
        )

    manifest_path = Path(args.eval_csv)
    summary = {
        "schema_version": OFFLINE_SCORE_SCHEMA_VERSION,
        "privacy_notice": PRIVACY_NOTICE,
        "evaluation_manifest": {
            "filename": manifest_path.name,
            "sha256": _sha256(manifest_path),
            "num_samples": len(next(iter(scores.values())).image_ids),
        },
        "segmentation_thresholds": {
            "od": float(segmentation_thresholds[0])
            if isinstance(segmentation_thresholds, tuple)
            else float(segmentation_thresholds),
            "oc": float(segmentation_thresholds[1])
            if isinstance(segmentation_thresholds, tuple)
            else float(segmentation_thresholds),
        },
        "methods": {
            method: {
                "prediction_artifact_filename": methods[method].name,
                "artifact_metadata": score.artifact_metadata,
                "metrics": score.metrics,
                "metric_definitions": score.metric_definitions,
            }
            for method, score in scores.items()
        },
        "bootstrap": bootstrap,
    }
    _write_private_text(
        output_dir / "summary.json",
        json.dumps(_json_safe(summary), indent=2, allow_nan=False) + "\n",
    )

    private_records = combine_private_records(scores)
    records_text = "".join(
        json.dumps(_json_safe(record), allow_nan=False, separators=(",", ":")) + "\n"
        for record in private_records
    )
    _write_private_text(output_dir / "per_image_records.jsonl", records_text)
    _write_private_text(output_dir / "PRIVATE_OUTPUT_NOTICE.txt", PRIVACY_NOTICE + "\n")
    print(json.dumps(_json_safe(summary), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
