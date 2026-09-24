# CuPGeo

Core experimental code for **Beyond Containment: Cup-Preserving Nested Geometry for Source-Only Cross-Domain Optic Disc and Cup Segmentation**.

[中文说明](README.zh-CN.md) | [Experiment map](docs/EXPERIMENTS.md) | [Data preparation](docs/DATA.md) | [Provenance and release status](docs/PROVENANCE.md)

This standalone source bundle was prepared on 2026-09-22. Use **this directory** as the repository root, not its parent historical collection. It includes CP nesting, bounded VRA localization, moment-based ratio supervision, training, frozen prediction, offline scoring, component/SG configurations, and probability-based geometry analysis. `c3tta` is the preserved Python namespace for checkpoint compatibility; the paper recipes use no test-time adaptation.

No datasets, weights, private per-image records, server credentials, or remote launchers are bundled. Packaging does not reproduce paper scores. The source code is released under the [MIT License](LICENSE); upstream dependencies, datasets, and pretrained models retain their own terms.

## Install

Use Python 3.10+ and install the PyTorch/torchvision build appropriate for your hardware in an isolated environment. Then, from this directory:

```bash
python -m pip install -e '.[dev,analysis]'
python -m pytest -q
```

The dependency ranges are not a lockfile of the historical training environment. Obtain the required datasets separately using [DATA.md](docs/DATA.md). Configurations use relative paths:

- `data/fundus_dg`: processed REFUGE/RIGA data;
- `manifests/source_train.csv`, `manifests/source_val.csv`: REFUGE 320/80 split;
- `weights/dinov3/model.safetensors`: separately obtained timm-compatible DINOv3-L/16 weights.

The historical backbone SHA256 is `45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941`. An arbitrary newer upstream download does not establish checkpoint identity. Dataset access, weight access, and their terms are the user's responsibility.

## Training protocols

Keep single-stage comparisons separate from matched component/SG ablations. Each stage allows 120 epochs and selects `best.pt` by source-validation Mean Dice; target labels are not used for selection.

```bash
# PRINT commands only; no training, dependencies, or data required.
python -m scripts.plan_experiments --suite single --seeds 0 1 2
python -m scripts.plan_experiments --suite matched --seeds 0 1 2
```

The matched plan has one CP pretraining run per seed, followed by five stage-2 runs: CP-only continuation, full CuPGeo, w/o ratio, w/o VRA, and full w/o SG. Every variant starts from the **same per-seed stage-1 CP checkpoint**, not the preceding variant. SG is disabled only in stage 2. `--init-checkpoint` restarts the optimizer, schedule, and warmup; `--resume` has a different purpose.

To run a prepared plan, explicitly add `--execute`. The helper runs jobs sequentially and refuses to overwrite existing output directories. For example, after preparing data and weights:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.plan_experiments \
  --suite matched --seeds 0 1 2 --execute
```

Use `--variants full no_ratio no_vra cp` to omit the SG comparison. For interrupted jobs use the original `scripts.train_source --resume` entry point with the corresponding config/output; the planner intentionally does not guess recovery state. See [the experiment map](docs/EXPERIMENTS.md) for direct commands and ratio-weight settings.

## Frozen prediction and offline scoring

Example for BinRushed after matched training:

```bash
python -m scripts.evaluate --config configs/cupgeo.yaml \
  --checkpoint runs/paper/matched/seed0/stage2_full/best.pt \
  --csv manifests/fundus_dg/binrushed_test.csv --data-root data/fundus_dg \
  --predictions runs/paper/matched/seed0/stage2_full/binrushed.predictions.pt \
  --artifact-method CuPGeo --precision fp16

python -m scripts.score_predictions \
  --prediction CuPGeo=runs/paper/matched/seed0/stage2_full/binrushed.predictions.pt \
  --eval-csv manifests/fundus_dg/binrushed_test.csv --data-root data/fundus_dg \
  --output-dir runs/paper/matched/seed0/stage2_full/scored/binrushed \
  --segmentation-threshold 0.5
```

`--predictions` uses the image-only loader even if the CSV also has mask columns. Labels are read separately by the scorer. Repeat for all four target domains and all prespecified seeds. Do not enable multi-scale/flip inference for the paper protocol. Keep private prediction and scoring artifacts under ignored `runs/` storage.

## Geometry analysis

The archived analysis utility is included unchanged as `scripts/analyze_ratio_geometry.py`. It reads frozen 768-by-768 prediction artifacts on CPU, exports probabilities, computes ratio/diameter errors, and supports moment-proxy statistics:

```bash
python -m scripts.analyze_ratio_geometry --spec configs/geometry_jobs.example.json \
  --source-root . --output runs/geometry_analysis --compute-proxy
```

The JSON is an **example**, not actual experiment output: supply the artifacts/manifests from your runs. It illustrates one complete four-domain seed; include every planned seed and method for paper-level mean and sample SD. The script exports FP32 probabilities from stored logits; it does not recover precision already lost in FP16 storage. The common scorer remains the primary source of reported segmentation/hard-vCDR metrics.

## Checks

```bash
python -m scripts.check_release
python -m unittest discover -s tests -p test_release_plan.py
python -m pytest -q
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -m scripts.smoke_test
```

The first command checks syntax, portable configs, checksums, and excluded artifact types. The smoke workflow uses synthetic images and a tiny backbone, exercises CP-to-CuPGeo initialization and scoring, and **does not reproduce paper results**. It performs a small amount of CPU training only when explicitly run. Current packaging checks are documented in [VALIDATION.md](docs/VALIDATION.md).
