<p align="center"><img src="assets/cupgeo-mark.svg" alt="CuPGeo mark" width="56" height="56"></p>
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

## <img src="assets/icon-method.svg" alt="" width="22" height="22"> Method

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

## <img src="assets/icon-start.svg" alt="" width="22" height="22"> Quick start

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

## <img src="assets/icon-protocol.svg" alt="" width="22" height="22"> Paper protocol

**Training data.** REFUGE supplies 320 training images and 80 held-out images for CuPGeo checkpoint and hyperparameter selection. The four target evaluation sets are BinRushed (39), Magrabia (19), RIM-ONE DL (174), and PAPILA (84). CuPGeo uses no target image or label for training or model selection. [Prepare the exact splits](docs/DATA.md) before launching a run.

| Manuscript experiment | Initialization and budget | Code path |
| :--- | :--- | :--- |
| Main comparison (Table 1) | Separate 120-epoch source runs for CuPGeo, MixStyle, and DSU | `--suite single` |
| Component ablation (Table 2) and additional SG control | CP pretraining for 120 epochs; independent 120-epoch continuations from the **same-seed CP best checkpoint**, including CP-only | `--suite matched` |
| Soft-vCDR weight study (source validation) | Same matched CP initialization; seven weights from 0.25 to 3.00, with 2.00 supplied by the full-model run | `--suite matched --reuse-cp --variants ratio_025 ...` |

These in-repository model runs use seeds **0, 1, 2**, 768×768 inputs, frozen DINOv3-L/16 with rank-8 QKV LoRA in the last four blocks, a Pyramid-FPN decoder, fp16, batch size 4, four-step accumulation, and AdamW. Decoder/LoRA learning rates are 2.5×10⁻⁴ / 5×10⁻⁵, with weight decay 10⁻⁴, five warmup epochs, and cosine decay to 10% of the initial rate. The source-validation **Mean Dice** selects `best.pt` in every stage. [Full hyperparameters and ablation switches](docs/EXPERIMENTS.md) are specified in the configs.

After preparing data and the separately obtained [DINOv3 checkpoint](docs/DATA.md), inspect the training commands; add `--execute` to run them:

~~~bash
python -m scripts.plan_experiments --suite single --seeds 0 1 2
python -m scripts.plan_experiments --suite matched --seeds 0 1 2
~~~

The matched plan trains one CP initializer per seed, then launches **CP-only, full, w/o ratio, w/o VRA, and full w/o SG** separately from it. For the six additional ratio weights, use `--reuse-cp` after matched CP checkpoints exist; [the exact command](docs/EXPERIMENTS.md#source-validation-weight-study) avoids repeating CP pretraining. `--init-checkpoint` loads model weights for a fresh stage; `--resume` restores an interrupted stage, including optimizer state.

### Frozen prediction → offline scoring

The predictor reads images without target labels. The common scorer reads labels only after the prediction artifact has been saved.

~~~bash
python -m scripts.evaluate --config configs/cupgeo.yaml \
  --checkpoint runs/paper/matched/seed0/stage2_full/best.pt \
  --csv manifests/fundus_dg/binrushed_test.csv --data-root data/fundus_dg \
  --predictions runs/paper/matched/seed0/stage2_full/predictions/binrushed.pt \
  --artifact-method full --precision fp16

python -m scripts.score_predictions \
  --prediction full=runs/paper/matched/seed0/stage2_full/predictions/binrushed.pt \
  --eval-csv manifests/fundus_dg/binrushed_test.csv --data-root data/fundus_dg \
  --output-dir runs/paper/matched/seed0/stage2_full/scored/binrushed \
  --segmentation-threshold 0.5
~~~

The paper uses a shared 0.5 threshold and no multi-scale/flip inference. Keep predictions and per-image scores under ignored <code>runs/</code> storage. For moment-proxy and diameter analysis, use [<code>scripts/analyze_ratio_geometry.py</code>](scripts/analyze_ratio_geometry.py) with the [example job specification](configs/geometry_jobs.example.json).

Score every seed on all four targets, then run [<code>scripts/aggregate_paper_domains.py</code>](scripts/aggregate_paper_domains.py). It checks the complete seed/domain matrix, checkpoint and manifest identities, source-only inference settings, and valid vCDR counts. It averages domains **equally within each seed**, then reports the cross-seed mean and **sample** SD. [End-to-end evaluation commands](docs/EXPERIMENTS.md#frozen-four-domain-evaluation) cover every target.

## <img src="assets/icon-map.svg" alt="" width="22" height="22"> Repository map

| Path | Role |
| :--- | :--- |
| [<code>c3tta/models/</code>](c3tta/models) | DINOv3 + LoRA backbone, Pyramid-FPN, VRA, and nested output construction |
| [<code>c3tta/losses/</code>](c3tta/losses) | OD/OC segmentation, allocation, soft-vCDR, and shared auxiliary losses |
| [<code>c3tta/data/</code>](c3tta/data) | Data loading, canonical masks, paired views, and geometry targets |
| [<code>configs/</code>](configs) | Full model, ablations, shared-backbone controls, and ratio-weight settings |
| [<code>scripts/</code>](scripts) | Data conversion, training plans, frozen prediction, scoring, paper aggregation, and geometry analysis |
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
