"""Print reproducible training commands; execute only with explicit --execute.

This packaging helper does not change the archived training implementation.
Run from the release root. A matched suite trains one CP initializer per seed,
then initializes EVERY stage-2 variant from that same checkpoint (not from
the preceding variant). Single-stage comparisons remain a separate suite.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PAPER_BACKBONE_SHA256 = "45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941"
CONFIGS = {
    "cp": "cp_baseline.yaml",
    "full": "cupgeo.yaml",
    "no_ratio": "cupgeo_no_ratio.yaml",
    "no_vra": "cupgeo_no_vra.yaml",
    "no_sg": "cupgeo_no_sg.yaml",
    "mixstyle": "mixstyle.yaml",
    "dsu": "dsu.yaml",
    "ratio_025": "cupgeo_ratio_025.yaml",
    "ratio_050": "cupgeo_ratio_050.yaml",
    "ratio_100": "cupgeo_ratio_100.yaml",
    "ratio_150": "cupgeo_ratio_150.yaml",
    "ratio_250": "cupgeo_ratio_250.yaml",
    "ratio_300": "cupgeo_ratio_300.yaml",
}
MATCHED = ("cp", "full", "no_ratio", "no_vra", "no_sg")
RATIO = ("ratio_025", "ratio_050", "ratio_100", "ratio_150", "full", "ratio_250", "ratio_300")


def verify_paper_inputs(overrides: dict[str, str]) -> None:
    """Fail before training if source splits or backbone differ from the paper."""
    def path_for(option: str, default: str) -> Path:
        path = Path(overrides.get(option, default))
        return path if path.is_absolute() else ROOT / path

    data_root = path_for("data-root", "data/fundus_dg")
    if not data_root.is_dir():
        raise ValueError(f"Missing source data root: {data_root}")
    id_sets = []
    for option, default, count in (("train-manifest", "manifests/source_train.csv", 320),
                                    ("val-manifest", "manifests/source_val.csv", 80)):
        path = path_for(option, default)
        if not path.is_file():
            raise ValueError(f"Missing source manifest: {path}")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        ids = [row.get("image_id", "") for row in rows]
        if len(rows) != count or any(not image_id for image_id in ids) or len(set(ids)) != count:
            raise ValueError(f"{path} must contain {count} unique image IDs")
        if any(row.get("source_domain") not in (None, "", "REFUGE") for row in rows):
            raise ValueError(f"{path} contains a non-REFUGE source row")
        id_sets.append(set(ids))
    if id_sets[0] & id_sets[1]:
        raise ValueError("Source training and validation image IDs overlap")
    checkpoint = path_for("backbone-checkpoint", "weights/dinov3/model.safetensors")
    if not checkpoint.is_file():
        raise ValueError(f"Missing DINOv3 checkpoint: {checkpoint}")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != PAPER_BACKBONE_SHA256:
        raise ValueError(f"DINOv3 checkpoint SHA256 differs from the paper initialization: {checkpoint}")


@dataclass(frozen=True)
class Job:
    name: str
    output: Path
    command: tuple[str, ...]


def build_plan(suite: str, seeds: list[int], variants: list[str],
               output: Path, overrides: dict[str, str] | None = None,
               reuse_cp: bool = False) -> list[Job]:
    if suite not in ("matched", "single"):
        raise ValueError("suite must be matched or single")
    if reuse_cp and suite != "matched":
        raise ValueError("--reuse-cp applies only to the matched suite")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("Use unique, non-negative seeds")
    allowed = MATCHED + tuple(variant for variant in RATIO if variant not in MATCHED) if suite == "matched" else ("full", "mixstyle", "dsu", "cp", "no_ratio", "no_vra")
    if not variants or len(set(variants)) != len(variants) or any(v not in allowed for v in variants):
        raise ValueError(f"Use unique variants from {allowed}")
    jobs = []

    def add(name: str, variant: str, seed: int, dest: Path,
            init: Path | None = None) -> None:
        command = [sys.executable, "-m", "scripts.train_source", "--config",
                   str(ROOT / "configs" / CONFIGS[variant]), "--seed", str(seed),
                   "--epochs", "120", "--output", str(dest)]
        if init is not None:
            command += ["--init-checkpoint", str(init)]
        for option, value in (overrides or {}).items():
            command += [f"--{option}", value]
        jobs.append(Job(name, dest, tuple(command)))

    for seed in seeds:
        base = output / suite / f"seed{seed}"
        init = None
        if suite == "matched":
            stage1 = base / "stage1_cp"
            if not reuse_cp:
                add(f"seed{seed}/stage1_cp", "cp", seed, stage1)
            init = stage1 / "best.pt"
        for variant in variants:
            name = f"stage2_{variant}" if suite == "matched" else variant
            add(f"seed{seed}/{name}", variant, seed, base / name, init)
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("matched", "single"), default="matched")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--variants", nargs="+", choices=tuple(CONFIGS))
    parser.add_argument("--output", type=Path, default=Path("runs/paper"))
    for option in ("data-root", "train-manifest", "val-manifest", "backbone-checkpoint"):
        parser.add_argument(f"--{option}")
    parser.add_argument("--execute", action="store_true", help="Actually run training sequentially; default prints only")
    parser.add_argument("--reuse-cp", action="store_true", help="Use existing same-seed stage1_cp/best.pt; do not retrain CP")
    args = parser.parse_args()
    variants = args.variants or (list(MATCHED) if args.suite == "matched" else ["full", "mixstyle", "dsu"])
    overrides = {option: str(Path(value).expanduser().resolve())
                 for option in ("data-root", "train-manifest", "val-manifest", "backbone-checkpoint")
                 if (value := getattr(args, option.replace("-", "_"))) is not None}
    try:
        jobs = build_plan(args.suite, args.seeds, variants, args.output.resolve(), overrides,
                          reuse_cp=args.reuse_cp)
    except ValueError as error:
        parser.error(str(error))
    for job in jobs:
        print(f"# {job.name}\n{shlex.join(job.command)}", flush=True)
    if not args.execute:
        print(f"# Dry run: {len(jobs)} commands; no files written and no training started.")
        return
    if Path.cwd().resolve() != ROOT:
        parser.error("Run --execute from the release root so config-relative paths resolve correctly")
    try:
        verify_paper_inputs(overrides)
    except ValueError as error:
        parser.error(str(error))
    if args.reuse_cp:
        missing_cp = [str(args.output.resolve() / "matched" / f"seed{seed}" / "stage1_cp" / "best.pt")
                      for seed in args.seeds
                      if not (args.output.resolve() / "matched" / f"seed{seed}" / "stage1_cp" / "best.pt").is_file()]
        if missing_cp:
            parser.error("Missing same-seed CP checkpoint(s): " + ", ".join(missing_cp))
    occupied = [str(job.output) for job in jobs if job.output.exists()]
    if occupied:
        parser.error("Refusing to overwrite existing runs: " + ", ".join(occupied))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for job in jobs:
        subprocess.run(job.command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
