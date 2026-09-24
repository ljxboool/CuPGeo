"""Plan frozen prediction and offline scoring on the paper's four targets.

Dry run by default. With --execute, validate target manifest counts and run
image-only prediction before any label-reading scorer for each target.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys

from scripts.aggregate_paper_domains import DOMAINS
from scripts.plan_experiments import CONFIGS, MATCHED, RATIO, ROOT


DATA = {
    "binrushed": ("manifests/fundus_dg/binrushed_test.csv", "data/fundus_dg"),
    "magrabia": ("manifests/fundus_dg/magrabia_test.csv", "data/fundus_dg"),
    "rim_one": ("manifests/rim_one_eval.csv", "data/rim_one"),
    "papila": ("manifests/papila_eval.csv", "data/papila"),
}


@dataclass(frozen=True)
class EvaluationJob:
    name: str
    checkpoint: Path
    manifest: Path
    prediction: Path
    summary: Path
    predict_command: tuple[str, ...]
    score_command: tuple[str, ...]


def build_plan(suite: str, seeds: list[int], variants: list[str],
               output: Path, precision: str = "fp16") -> list[EvaluationJob]:
    if suite not in ("single", "matched"):
        raise ValueError("suite must be single or matched")
    allowed = ("mixstyle", "dsu") if suite == "single" else MATCHED + tuple(
        variant for variant in RATIO if variant not in MATCHED)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Use unique, non-negative seeds")
    if not variants or len(set(variants)) != len(variants) or any(v not in allowed for v in variants):
        raise ValueError(f"Use unique variants from {allowed}")
    if precision not in ("fp16", "bf16", "fp32"):
        raise ValueError("Unsupported precision")
    jobs = []
    for seed in seeds:
        for variant in variants:
            run = output / suite / f"seed{seed}" / (
                f"stage2_{variant}" if suite == "matched" else variant)
            checkpoint = run / "best.pt"
            config = ROOT / "configs" / CONFIGS[variant]
            for domain in DOMAINS:
                csv_path, data_root = DATA[domain]
                prediction = run / "predictions" / f"{domain}.pt"
                summary = run / "scored" / domain / "summary.json"
                predict = (sys.executable, "-m", "scripts.evaluate", "--config", str(config),
                           "--checkpoint", str(checkpoint), "--csv", csv_path,
                           "--data-root", data_root, "--predictions", str(prediction),
                           "--artifact-method", variant, "--precision", precision)
                score = (sys.executable, "-m", "scripts.score_predictions", "--prediction",
                         f"{variant}={prediction}", "--eval-csv", csv_path,
                         "--data-root", data_root, "--output-dir", str(summary.parent),
                         "--segmentation-threshold", "0.5", "--bootstrap-samples", "0")
                jobs.append(EvaluationJob(f"{suite}/seed{seed}/{variant}/{domain}", checkpoint,
                                          Path(csv_path), prediction, summary, predict, score))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("single", "matched"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--variants", nargs="+", choices=tuple(CONFIGS))
    parser.add_argument("--output", type=Path, default=Path("runs/paper"))
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    variants = args.variants or (["mixstyle", "dsu"] if args.suite == "single" else list(MATCHED))
    try:
        jobs = build_plan(args.suite, args.seeds, variants, args.output.resolve(), args.precision)
    except ValueError as exc:
        parser.error(str(exc))
    for job in jobs:
        print(f"# {job.name}\n{shlex.join(job.predict_command)}\n{shlex.join(job.score_command)}", flush=True)
    if not args.execute:
        print(f"# Dry run: {len(jobs)} target evaluations; no files written.")
        return
    if Path.cwd().resolve() != ROOT:
        parser.error("Run --execute from the release root")
    if (ROOT / "runs").resolve() not in args.output.resolve().parents:
        parser.error("Paper evaluation output must be below ignored runs/")
    for domain, (filename, expected_count) in DOMAINS.items():
        path = ROOT / DATA[domain][0]
        if not path.is_file() or path.name != filename:
            parser.error(f"Missing paper target manifest: {path}")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            count = sum(1 for _ in csv.DictReader(handle))
        if count != expected_count:
            parser.error(f"{path} has {count} rows; expected {expected_count}")
    for job in jobs:
        if not job.checkpoint.is_file():
            parser.error(f"Missing selected checkpoint: {job.checkpoint}")
        if job.prediction.exists() or job.summary.exists():
            parser.error(f"Refusing to overwrite existing evaluation: {job.name}")
    for job in jobs:
        subprocess.run(job.predict_command, cwd=ROOT, check=True)
        subprocess.run(job.score_command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
