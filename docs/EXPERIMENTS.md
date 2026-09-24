# Experiment-to-code map

This document supersedes the parent collection's 2026-09-20 single-stage-only ablation description. It refers to the revised manuscript's component/SG protocol, not every historical run stored in the workspace.

| Experiment | Config | Initialization / budget |
| --- | --- | --- |
| CP pretraining | `configs/cp_baseline.yaml` | Pretrained backbone, 120 epochs |
| CP / w/o VRA & ratio | `configs/cp_baseline.yaml` | Same-seed CP best, another 120 epochs |
| Full CuPGeo | `configs/cupgeo.yaml` | Same-seed CP best, another 120 epochs |
| w/o ratio | `configs/cupgeo_no_ratio.yaml` | Same-seed CP best, another 120 epochs |
| w/o VRA / CP + ratio | `configs/cupgeo_no_vra.yaml` | Same-seed CP best, another 120 epochs |
| Full w/o SG | `configs/cupgeo_no_sg.yaml` | Same-seed CP best, SG disabled only in stage 2 |
| Shared-backbone single-stage comparison | `cupgeo.yaml`, `mixstyle.yaml`, `dsu.yaml` | Separate 120-epoch runs; do not substitute matched-stage results |
| Ratio weights | `cupgeo_ratio_025/050/100/150/250/300.yaml`, `cupgeo.yaml` for 2.00 | Seven settings; keep the same prespecified seeds and initialization protocol |

All configurations specify one 120-epoch stage. The launch command determines whether it is source training from the pretrained backbone or a second stage initialized from CP. The same YAML alone does not prove the provenance of a reported result.

## Direct matched commands (seed 0)

```bash
python -m scripts.train_source --config configs/cp_baseline.yaml \
  --seed 0 --output runs/matched/seed0/stage1_cp

python -m scripts.train_source --config configs/cupgeo.yaml \
  --seed 0 --init-checkpoint runs/matched/seed0/stage1_cp/best.pt \
  --output runs/matched/seed0/stage2_full

python -m scripts.train_source --config configs/cupgeo_no_sg.yaml \
  --seed 0 --init-checkpoint runs/matched/seed0/stage1_cp/best.pt \
  --output runs/matched/seed0/stage2_no_sg
```

Use the same initialization for `cp_baseline.yaml`, `cupgeo_no_ratio.yaml`, and `cupgeo_no_vra.yaml` in separate stage-2 directories. Repeat for seeds 1 and 2. `--init-checkpoint` loads model state only; `--resume` restores the running job state and must not replace it for a fresh second stage.

## Scope of each ablation

- **w/o VRA:** disable the geometry head, logit calibration, and allocation loss; retain ratio weight 2, CP, SG, paired-scale supervision, and anatomy-field objectives.
- **w/o ratio:** retain CP, SG and VRA; set ratio loss weight to zero.
- **CP-only:** disable VRA and ratio, retaining shared objectives and a full matched second stage.
- **w/o SG:** change only `model.detach_cup_from_od` in full CuPGeo's second stage. Forward probabilities and other loss weights remain unchanged.

## Metrics and aggregation

Primary evaluation is `scripts.score_predictions` with threshold 0.5 on both masks. Scores include OD/OC/Mean Dice, hard-mask vCDR MAE, and `topology_violation` (the **proportion of images** with any cup pixel outside the disc). Multiply this proportion by 100 to display CVR in percent. The scorer records empty-disc/invalid-ratio cases; do not silently replace missing values with zero.

Average the four domains equally **within each seed**, then report the mean and sample SD (`ddof=1`) across all prespecified seeds. Do not pool images across domains, average domain SDs, or choose seeds based on target scores. `aggregate_source_method_seeds.py` is a strict per-domain cross-seed comparison utility, not an automatic four-domain macro aggregator. `analyze_ratio_geometry.py` provides geometry statistics and four-target macro aggregation for a complete job matrix; a single-seed example is not a three-seed paper result.

The weight-selection study uses the REFUGE 80-image held-out source split, also used for checkpoint selection. It must stay separate from target-domain results. Historical weight-study values, seed selection issues, and third-party run provenance are not relabeled as verified by this source bundle; independently audit the exact launch/checkpoint records before claiming full table reproduction.

## Third-party baselines

MixStyle and DSU shared-backbone configurations are included. Other authors' standalone implementations are not re-vendored into this clean bundle. The parent historical collection retains available external snapshots and their audit notes; original licenses, dependencies, and method-specific protocols must be checked before redistributing them. This package is the core CuPGeo experimental code, not a claim that every external baseline or manuscript table has been fully reproduced.
