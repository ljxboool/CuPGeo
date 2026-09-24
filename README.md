<h1 align="center">CuPGeo</h1>
<p align="center">
  <strong>Beyond Containment</strong><br>
  Cup-Preserving Nested Geometry for Source-Only Cross-Domain<br>
  Optic Disc and Cup Segmentation
</p>

<p align="center"><sub>Source-only domain generalization · MIT-licensed research code</sub></p>

<p align="center">
  <a href="#method">Method</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#paper-protocol">Paper protocol</a> ·
  <a href="#repository-map">Code map</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

---

## Method

CuPGeo treats the optic cup as a geometric anchor. A vertical region allocation (VRA) head provides a bounded cup-localization prior; cup-preserving (CP) composition completes the optic disc with residual rim probability; and a differentiable soft-vCDR objective supervises their relative vertical extent. The model trains on labeled **source** images and performs frozen inference on unseen target domains.

<p align="center">
  <img src="assets/cupgeo-method.svg" alt="CuPGeo method: shared features, vertical allocation, cup-preserving composition, and soft-vCDR supervision" width="100%">
</p>

| Geometry | What it contributes |
| :--- | :--- |
| **Vertical allocation** | A five-region vertical prior applies a bounded correction where cup predictions are uncertain. |
| **Cup-preserving composition** | The disc is assembled from the cup and residual rim, giving pixelwise OD probability at least as large as OC probability. |
| **Soft vCDR** | Moment-based vertical spreads supply differentiable ratio supervision during source training. |

The central composition is **P<sub>OD</sub> = P<sub>OC</sub> + (1 − P<sub>OC</sub>) P<sub>rim</sub>**. A shared threshold therefore produces nested OD/OC masks. Stop-gradient changes the *training gradient path* through the disc side; it does not change the forward probabilities. There is no test-time adaptation or morphological post-processing in the paper protocol.

## Quick start

The commands below install the package, validate the source release, and **print** the matched experiment plan. The planner does not train unless <code>--execute</code> is supplied.

~~~bash
git clone https://github.com/ljxboool/CuPGeo.git
cd CuPGeo

# Install a PyTorch/torchvision build appropriate for your hardware first.
python -m pip install -e '.[dev,analysis]'
python -m scripts.check_release
python -m scripts.plan_experiments --suite matched --seeds 0 1 2
~~~

Datasets and the DINOv3-L/16 backbone are obtained separately. See [data preparation](docs/DATA.md) and the [experiment map](docs/EXPERIMENTS.md) before executing training. The original backbone SHA-256 is <code>45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941</code>; a different upstream revision is not the identical initialization.

## Paper protocol

| Experiment family | Initialization | Training budget | Checkpoint selection |
| :--- | :--- | :--- | :--- |
| Shared-backbone comparisons | Pretrained backbone | 120 epochs | Source-validation Mean Dice |
| Component and stop-gradient ablations | Same-seed CP checkpoint after 120 epochs | Separate 120-epoch continuation for each variant, including CP-only | Source-validation Mean Dice |

The matched planner schedules CP pretraining once per seed, then starts **CP-only**, **full CuPGeo**, **w/o ratio**, **w/o VRA**, and **full w/o SG** independently from that same checkpoint. It never chains one ablation into another. Target labels are used only after frozen prediction, for evaluation. The four reported target domains are **BinRushed**, **Magrabia**, **RIM-ONE DL**, and **PAPILA**; four-domain scores average domains within each seed before computing cross-seed mean and sample SD.

After preparing data and weights, run the plan explicitly:

~~~bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.plan_experiments \
  --suite matched --seeds 0 1 2 --execute
~~~

For a direct source-training or resumed run, use [<code>scripts/train_source.py</code>](scripts/train_source.py). <code>--init-checkpoint</code> begins a fresh second stage with model weights only; <code>--resume</code> restores an interrupted job. See [the experiment map](docs/EXPERIMENTS.md) for direct commands, ablation switches, and ratio-weight configurations.

### Frozen prediction → offline scoring

The predictor reads images without target labels. The common scorer reads labels only after the prediction artifact has been saved.

~~~bash
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
~~~

The paper uses a shared 0.5 threshold and no multi-scale/flip inference. Keep predictions and per-image scores under ignored <code>runs/</code> storage. For moment-proxy and diameter analysis, use [<code>scripts/analyze_ratio_geometry.py</code>](scripts/analyze_ratio_geometry.py) with the [example job specification](configs/geometry_jobs.example.json).

## Repository map

| Path | Role |
| :--- | :--- |
| [<code>c3tta/models/</code>](c3tta/models) | DINOv3 + LoRA backbone, Pyramid-FPN, VRA, and nested output construction |
| [<code>c3tta/losses/</code>](c3tta/losses) | OD/OC segmentation, allocation, soft-vCDR, and shared auxiliary losses |
| [<code>c3tta/data/</code>](c3tta/data) | Data loading, canonical masks, paired views, and geometry targets |
| [<code>configs/</code>](configs) | Full model, ablations, shared-backbone controls, and ratio-weight settings |
| [<code>scripts/</code>](scripts) | Data conversion, training, frozen prediction, scoring, and analysis |
| [<code>docs/</code>](docs) | [Data](docs/DATA.md) · [experiments](docs/EXPERIMENTS.md) · [provenance](docs/PROVENANCE.md) · [validation](docs/VALIDATION.md) |

The historical <code>c3tta</code> Python namespace is retained for checkpoint compatibility. The current paper recipes use source-only training.

## Release scope

This repository contains **code, configurations, tests, and documentation**. It does **not** contain retinal datasets, pretrained or trained weights, per-seed checkpoints, predictions, score files, credentials, or server launch scripts. The release checker verifies syntax, configuration consistency, source hashes, and the source-only file inventory; it is not a claim that publishing the package reproduced the paper's numerical results.

~~~bash
python -m scripts.check_release
python -m unittest discover -s tests -p test_release_plan.py
python -m pytest -q
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -m scripts.smoke_test
~~~

The smoke test uses synthetic images and a tiny backbone. Dataset access and pretrained-model terms remain with their respective providers. The source code is available under the [MIT License](LICENSE). For attribution, see [<code>CITATION.cff</code>](CITATION.cff); publication details will be added when available.
