# Experiment-to-code map

This is the executable map for the current manuscript. Run commands from the repository root. Tables 1 and 2 report **240-epoch** results. Table 1 CuPGeo and Table 2 Full use the same matched Full run (CP 120 + Full 120); the in-repository MixStyle and DSU baselines each use a separate 240-epoch run from the pretrained backbone.

| Experiment | Config | Initialization / budget |
| --- | --- | --- |
| CP pretraining | `configs/cp_baseline.yaml` | Pretrained backbone, 120 epochs |
| CP / w/o VRA & ratio | `configs/cp_baseline.yaml` | Same-seed CP best, another 120 epochs |
| Full CuPGeo | `configs/cupgeo.yaml` | Same-seed CP best, another 120 epochs |
| w/o ratio | `configs/cupgeo_no_ratio.yaml` | Same-seed CP best, another 120 epochs |
| w/o VRA / CP + ratio | `configs/cupgeo_no_vra.yaml` | Same-seed CP best, another 120 epochs |
| Full w/o SG | `configs/cupgeo_no_sg.yaml` | Same-seed CP best, SG disabled only in stage 2 |
| Table 1: CuPGeo | `cupgeo.yaml` | Reuse matched `stage2_full/best.pt` after CP 120 + Full 120 |
| Table 1: MixStyle / DSU | `mixstyle.yaml`, `dsu.yaml` | Separate 240-epoch runs from the pretrained backbone |
| Source weight table | `cupgeo_ratio_025/050/100/150/250/300.yaml`, `cupgeo.yaml` for 2.00 | Seven settings, from the same-seed CP checkpoint; source validation only |
| Ratio-proxy table | `scripts/analyze_ratio_geometry.py` | Reuse frozen Full and w/o ratio predictions; no further training |

CuPGeo component configs specify one 120-epoch stage; the MixStyle and DSU configs specify 240 epochs. The launch command determines whether a CuPGeo config is CP pretraining or a second stage initialized from CP. The same YAML alone does not prove the provenance of a reported result.

## Fixed conditions

| Item | Paper setting |
| --- | --- |
| Source | REFUGE train 320; held-out source validation 80 for checkpoint and hyperparameter selection |
| Targets | BinRushed 39; Magrabia 19; RIM-ONE DL hospital test 174; PAPILA patient-level test 84 |
| Model | Frozen DINOv3-L/16; features from blocks 6/12/18/24; last four blocks QKV LoRA rank 8, alpha 16; Pyramid-FPN decoder |
| Input and compute | 768×768; fp16; batch size 4; gradient accumulation 4; seeds 0/1/2 |
| Optimizer | AdamW; decoder/LoRA LR 2.5e-4/5e-5; weight decay 1e-4; five warmup epochs and cosine decay to 10% |
| Checkpoint | Maximize source-validation Mean Dice (`val_seg_dice`) for every stage or 240-epoch baseline run |
| Full geometry | VRA weight .25, temperature .02, kappa 2, alpha initial 0; soft-vCDR weight 2, SmoothL1 transition .05; stop-gradient on disc side |
| Shared objectives (CuPGeo variants) | Paired log-uniform scale 0.2–1.0, center jitter .2; mask equivariance .50; anatomy field .15/.20/.10; only these auxiliary losses ramp over 12 epochs |
| Inference | Frozen checkpoint; single 768×768 view, no flip/multiscale ensemble or target adaptation; shared OD/OC threshold .5 |

The source mask drives BCE + Dice on OD and OC. VRA's five-region allocation and the moment-based soft-vCDR objective are active only where the configuration enables them. The `cp_baseline.yaml` branch retains the shared paired-scale/anatomy-field objectives. The stage-2 w/o SG variant changes the *training gradient path* only; the forward CP composition is unchanged.

Before running, create the [paper manifests and data roots](DATA.md), place the separately obtained backbone at `weights/dinov3/model.safetensors`, and verify its original SHA-256 (also recorded in [DATA.md](DATA.md)). The released code does not include datasets, weights, historical run outputs, or third-party implementations.

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

For the complete three-seed launch, inspect the dry-run plans and add `--execute` only when data and backbone weights are ready. Before execution, the planner checks the REFUGE 320/80 source split, disjoint image IDs, and the original DINOv3 SHA-256. It refuses to overwrite an existing run directory.

```bash
python -m scripts.plan_experiments --suite single --seeds 0 1 2
python -m scripts.plan_experiments --suite matched --seeds 0 1 2

CUDA_VISIBLE_DEVICES=0 python -m scripts.plan_experiments --suite single --seeds 0 1 2 --execute
CUDA_VISIBLE_DEVICES=0 python -m scripts.plan_experiments --suite matched --seeds 0 1 2 --execute
```

The `single` suite supplies the in-repository 240-epoch MixStyle and DSU runs. CuPGeo in Table 1 comes from the `matched` suite's Full checkpoint, also used in Table 2. Other Table 1 methods follow their own architectures and are not silently mapped to these configs. The `matched` suite performs one CP stage 1 per seed and five independent stage-2 runs per seed.

## Source-validation weight study

After `matched` stage 1 exists, reuse its three CP checkpoints for the six additional ratio weights; the already completed `stage2_full` supplies weight 2.00. Every weight is selected and compared on the same 80-image REFUGE source validation set, not on target domains.

```bash
python -m scripts.plan_experiments --suite matched --seeds 0 1 2 --reuse-cp \
  --variants ratio_025 ratio_050 ratio_100 ratio_150 ratio_250 ratio_300

CUDA_VISIBLE_DEVICES=0 python -m scripts.plan_experiments --suite matched --seeds 0 1 2 \
  --reuse-cp --variants ratio_025 ratio_050 ratio_100 ratio_150 ratio_250 ratio_300 --execute
```

The planner checks that each existing same-seed `stage1_cp/best.pt` is present before execution. Report the source-validation metrics saved by the selected checkpoint for each setting; do not pick a weight using target labels.

## Scope of each ablation

- **w/o VRA:** disable the geometry head, logit calibration, and allocation loss; retain ratio weight 2, CP, SG, paired-scale supervision, and anatomy-field objectives.
- **w/o ratio:** retain CP, SG and VRA; set ratio loss weight to zero.
- **CP-only:** disable VRA and ratio, retaining shared objectives and a full matched second stage.
- **w/o SG:** change only `model.detach_cup_from_od` in full CuPGeo's second stage. Forward probabilities and other loss weights remain unchanged.

## Frozen four-domain evaluation

The evaluation planner covers **every** selected seed and all four target manifests. It checks their expected paper counts before execution, runs `scripts.evaluate --predictions` without loading labels, then calls the common `scripts.score_predictions` offline scorer. Outputs stay below ignored `runs/`. It refuses to overwrite existing predictions or summaries.

```bash
python -m scripts.plan_paper_evaluation --suite matched --seeds 0 1 2
python -m scripts.plan_paper_evaluation --suite single --seeds 0 1 2

CUDA_VISIBLE_DEVICES=0 python -m scripts.plan_paper_evaluation --suite matched --seeds 0 1 2 --execute
CUDA_VISIBLE_DEVICES=0 python -m scripts.plan_paper_evaluation --suite single --seeds 0 1 2 --execute

python -m scripts.aggregate_paper_domains --suite matched --variant full --method full \
  --seeds 0 1 2 --output runs/paper/matched/full_macro4.json
python -m scripts.aggregate_paper_domains --suite single --variant mixstyle --method mixstyle \
  --seeds 0 1 2 --output runs/paper/single/mixstyle_macro4.json
```

The same aggregation command accepts any completed variant, for example `--variant no_vra --method no_vra` in the matched suite. Score summaries and per-image records are private evaluation artifacts. For ratio-proxy and diameter analyses, reuse the saved probability artifacts with `scripts.analyze_ratio_geometry` and `configs/geometry_jobs.example.json`.
Figure 3 uses fixed seed-0 cases from the matched full/ablation predictions, with the shared 0.5 threshold; the repository does not publish patient images or rendered masks.

## Metrics and aggregation

Primary evaluation is `scripts.score_predictions` with threshold 0.5 on both masks. Scores include OD/OC/Mean Dice, hard-mask vCDR MAE, and `topology_violation` (the **proportion of images** with any cup pixel outside the disc). Multiply this proportion by 100 to display CVR in percent. The scorer records empty-disc/invalid-ratio cases; do not silently replace missing values with zero. `aggregate_paper_domains.py` reports valid/total vCDR counts and rejects changed manifest hashes, checkpoints, thresholds, seeds, configs, or inference settings.

Average the four domains equally **within each seed**, then report the mean and sample SD (`ddof=1`) across all prespecified seeds. Do not pool images across domains, average domain SDs, or choose seeds based on target scores. `aggregate_paper_domains.py` implements this paper macro; `aggregate_source_method_seeds.py` remains a per-domain comparison utility. A single-seed geometry example is not a three-seed paper result.

The weight-selection study uses the REFUGE 80-image held-out source split, also used for checkpoint selection. It must stay separate from target-domain results. Historical weight-study values, seed selection issues, and third-party run provenance are not relabeled as verified by this source bundle; independently audit the exact launch/checkpoint records before claiming full table reproduction.

## Third-party baselines

MixStyle and DSU shared-backbone configurations are included. Other authors' standalone implementations are not re-vendored into this clean bundle. The parent historical collection retains available external snapshots and their audit notes; original licenses, dependencies, and method-specific protocols must be checked before redistributing them. This package is the core CuPGeo experimental code, not a claim that every external baseline or manuscript table has been fully reproduced.
