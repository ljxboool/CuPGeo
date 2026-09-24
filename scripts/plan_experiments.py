"""Print reproducible training commands; execute only with explicit --execute.

This packaging helper does not change the archived training implementation.
Run from the release root. A matched suite trains one CP initializer per seed,
then initializes EVERY stage-2 variant from that same checkpoint (not from
the preceding variant). Single-stage comparisons remain a separate suite.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "cp": "cp_baseline.yaml",
    "full": "cupgeo.yaml",
    "no_ratio": "cupgeo_no_ratio.yaml",
    "no_vra": "cupgeo_no_vra.yaml",
    "no_sg": "cupgeo_no_sg.yaml",
    "mixstyle": "mixstyle.yaml",
    "dsu": "dsu.yaml",
}
MATCHED = ("cp", "full", "no_ratio", "no_vra", "no_sg")


@dataclass(frozen=True)
class Job:
    name: str
    output: Path
    command: tuple[str, ...]


def build_plan(suite: str, seeds: list[int], variants: list[str],
               output: Path, overrides: dict[str, str] | None = None) -> list[Job]:
    if suite not in ("matched", "single"):
        raise ValueError("suite must be matched or single")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("Use unique, non-negative seeds")
    allowed = MATCHED if suite == "matched" else ("full", "mixstyle", "dsu", "cp", "no_ratio", "no_vra")
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
    args = parser.parse_args()
    variants = args.variants or (list(MATCHED) if args.suite == "matched" else ["full", "mixstyle", "dsu"])
    overrides = {option: str(Path(value).expanduser().resolve())
                 for option in ("data-root", "train-manifest", "val-manifest", "backbone-checkpoint")
                 if (value := getattr(args, option.replace("-", "_"))) is not None}
    try:
        jobs = build_plan(args.suite, args.seeds, variants, args.output.resolve(), overrides)
    except ValueError as error:
        parser.error(str(error))
    for job in jobs:
        print(f"# {job.name}\n{shlex.join(job.command)}", flush=True)
    if not args.execute:
        print(f"# Dry run: {len(jobs)} commands; no files written and no training started.")
        return
    if Path.cwd().resolve() != ROOT:
        parser.error("Run --execute from the release root so config-relative paths resolve correctly")
    occupied = [str(job.output) for job in jobs if job.output.exists()]
    if occupied:
        parser.error("Refusing to overwrite existing runs: " + ", ".join(occupied))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for job in jobs:
        subprocess.run(job.command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
