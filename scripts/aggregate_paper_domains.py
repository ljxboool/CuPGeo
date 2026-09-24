"""Aggregate the paper's four target domains within seed, then across seeds.

Read private ``score_predictions`` summaries from a completed paper run. This
script never selects a checkpoint or reads target images. It rejects incomplete
or mismatched evaluation protocols rather than pooling available rows.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Mapping

from scripts.plan_experiments import CONFIGS


DOMAINS = {
    "binrushed": ("binrushed_test.csv", 39),
    "magrabia": ("magrabia_test.csv", 19),
    "rim_one": ("rim_one_eval.csv", 174),
    "papila": ("papila_eval.csv", 84),
}
METRICS = ("od_dice", "oc_dice", "dice_mean", "vcdr_mae", "topology_violation")


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def aggregate(
    summaries: Mapping[tuple[int, str], Mapping[str, Any]], *,
    seeds: tuple[int, ...], method: str, variant: str, precision: str = "fp16",
) -> dict[str, Any]:
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Use at least two distinct seeds for sample SD")
    expected = {(seed, domain) for seed in seeds for domain in DOMAINS}
    if set(summaries) != expected:
        raise ValueError(f"Incomplete or extra seed/domain scores: missing={sorted(expected - set(summaries))}, extra={sorted(set(summaries) - expected)}")
    if variant not in CONFIGS:
        raise ValueError(f"Unknown variant: {variant}")

    per_seed: dict[str, Any] = {}
    manifest_hashes: dict[str, str] = {}
    checkpoint_hashes: dict[int, str] = {}
    for seed in seeds:
        domain_values: dict[str, dict[str, float | int]] = {}
        for domain, (filename, count) in DOMAINS.items():
            summary = summaries[(seed, domain)]
            if summary.get("schema_version") != "c3tta.offline_scores.v1":
                raise ValueError(f"seed {seed}/{domain}: unexpected score schema")
            manifest = summary.get("evaluation_manifest", {})
            if manifest.get("filename") != filename or manifest.get("num_samples") != count:
                raise ValueError(f"seed {seed}/{domain}: expected {count} images from {filename}")
            digest = manifest.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"seed {seed}/{domain}: missing manifest SHA256")
            if domain in manifest_hashes and manifest_hashes[domain] != digest:
                raise ValueError(f"seed {seed}/{domain}: manifest differs across seeds")
            manifest_hashes[domain] = digest
            if summary.get("segmentation_thresholds") != {"od": 0.5, "oc": 0.5}:
                raise ValueError(f"seed {seed}/{domain}: expected shared 0.5 threshold")

            result = summary.get("methods", {}).get(method)
            if not isinstance(result, dict):
                raise ValueError(f"seed {seed}/{domain}: missing method {method}")
            metadata = result.get("artifact_metadata", {})
            if (metadata.get("method") != method or metadata.get("source_seed") != seed
                    or metadata.get("config_filename") != CONFIGS[variant]
                    or metadata.get("checkpoint_selection_metric") != "val_seg_dice"
                    or metadata.get("inference_precision") != precision
                    or metadata.get("optimization_applied") is not False
                    or metadata.get("inference_target_labels_loaded") is not False
                    or metadata.get("inference_tta_image_sizes") != [768]
                    or metadata.get("inference_tta_horizontal_flip") is not False):
                raise ValueError(f"seed {seed}/{domain}: prediction does not match the paper protocol")
            checkpoint = metadata.get("checkpoint_sha256")
            if not isinstance(checkpoint, str) or len(checkpoint) != 64:
                raise ValueError(f"seed {seed}/{domain}: missing checkpoint SHA256")
            if seed in checkpoint_hashes and checkpoint_hashes[seed] != checkpoint:
                raise ValueError(f"seed {seed}/{domain}: different checkpoints across domains")
            checkpoint_hashes[seed] = checkpoint

            raw = result.get("metrics", {})
            values = {key: _number(raw.get(key), f"seed {seed}/{domain}/{key}") for key in METRICS}
            valid = _number(raw.get("vcdr_valid_samples"), f"seed {seed}/{domain}/vcdr_valid_samples")
            invalid = _number(raw.get("vcdr_invalid_predictions"), f"seed {seed}/{domain}/vcdr_invalid_predictions")
            if (valid < 0 or invalid < 0 or valid != int(valid) or invalid != int(invalid)
                    or valid + invalid != count):
                raise ValueError(f"seed {seed}/{domain}: invalid vCDR coverage")
            if any(not 0 <= values[key] <= 1 for key in ("od_dice", "oc_dice", "dice_mean", "topology_violation")):
                raise ValueError(f"seed {seed}/{domain}: segmentation metric outside [0, 1]")
            if values["vcdr_mae"] < 0:
                raise ValueError(f"seed {seed}/{domain}: negative vCDR MAE")
            if abs(values["dice_mean"] - (values["od_dice"] + values["oc_dice"]) / 2) > 1e-6:
                raise ValueError(f"seed {seed}/{domain}: Mean Dice does not match OD/OC")
            values["cvr_percent"] = 100 * values.pop("topology_violation")
            values["vcdr_valid_samples"] = int(valid)
            values["vcdr_total_samples"] = count
            domain_values[domain] = values

        macro = {
            key: statistics.fmean(float(domain_values[domain][key]) for domain in DOMAINS)
            for key in ("od_dice", "oc_dice", "dice_mean", "vcdr_mae", "cvr_percent")
        }
        per_seed[str(seed)] = {"domains": domain_values, "macro4": macro}

    if len(set(checkpoint_hashes.values())) != len(seeds):
        raise ValueError("Different seeds reuse the same checkpoint")
    summary_stats = {}
    for domain in (*DOMAINS, "macro4"):
        summary_stats[domain] = {}
        for key in ("od_dice", "oc_dice", "dice_mean", "vcdr_mae", "cvr_percent"):
            values = [float((per_seed[str(seed)]["macro4"] if domain == "macro4"
                             else per_seed[str(seed)]["domains"][domain])[key]) for seed in seeds]
            summary_stats[domain][key] = {"mean": statistics.fmean(values), "sample_sd": statistics.stdev(values)}
    return {
        "protocol": "source-only; four-domain equal-weight within seed; sample SD across seeds",
        "method": method, "variant": variant, "seeds": list(seeds),
        "manifest_sha256_by_domain": manifest_hashes,
        "checkpoint_sha256_by_seed": {str(k): v for k, v in checkpoint_hashes.items()},
        "per_seed": per_seed, "summary": summary_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/paper"))
    parser.add_argument("--suite", choices=("matched", "single"), required=True)
    parser.add_argument("--variant", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--method", required=True, help="Exact --artifact-method used for prediction")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--output", type=Path, default=None, help="Optional private JSON file below runs/")
    args = parser.parse_args()
    if args.suite == "single" and args.variant not in ("mixstyle", "dsu"):
        parser.error("The 240-epoch single-stage suite contains MixStyle and DSU only; CuPGeo uses matched Full")
    summaries = {}
    try:
        for seed in args.seeds:
            run = args.root / args.suite / f"seed{seed}" / (
                f"stage2_{args.variant}" if args.suite == "matched" else args.variant
            )
            for domain in DOMAINS:
                path = run / "scored" / domain / "summary.json"
                summaries[(seed, domain)] = json.loads(path.read_text(encoding="utf-8"))
        result = aggregate(summaries, seeds=tuple(args.seeds), method=args.method,
                           variant=args.variant, precision=args.precision)
        output = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            target = args.output.resolve()
            private_root = Path("runs").resolve()
            if private_root not in target.parents:
                raise ValueError("Output must be below ignored runs/")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(output, encoding="utf-8")
            os.chmod(target, 0o600)
        else:
            print(output, end="")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
