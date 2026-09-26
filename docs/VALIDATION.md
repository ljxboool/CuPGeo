# Packaging validation

This bundle is a source release, not a newly reproduced experiment. Publishing it does not run full training or verify the paper's numerical results.

The current manuscript states that Tables 1 and 2 use 240-epoch runs. This release maps their identical CuPGeo Full values to the same matched CP 120 + Full 120 checkpoint. MixStyle and DSU use separate 240-epoch runs; a single-stage 120-epoch CuPGeo run is not presented as Table 1 evidence.

## Checks performed

- Python AST syntax checking: all packaged Python files parsed successfully.
- Configuration inheritance: all 14 YAML configurations resolved. CuPGeo stages are 120 epochs; MixStyle/DSU baseline runs are 240 epochs. Inputs are 768 square, and data/weight locations are relative paths.
- Ablation isolation: w/o SG differs only in `model.detach_cup_from_od`; w/o VRA changes only the geometry head, calibration switch, allocation switch, and allocation weight. Ratio weights and remaining shared objectives are preserved.
- Dependency-free protocol tests cover common per-seed CP initialization, single-stage separation, CP reuse for the ratio sweep, the four-target evaluation matrix, and domain-before-seed aggregation.
- Dry-run command generation passed: matched seeds 0/1/2 produce 18 training commands (3 initializers and 15 stage-2 jobs); the default 240-epoch baseline set produces 6. A six-weight ratio sweep with CP reuse produces 18 further stage-2 commands. The four-target full-model evaluation plan produces 12 prediction/scoring pairs. No training or real-data evaluation commands were executed by these checks.
- Focused CPU checks passed for CP, full, w/o ratio, w/o VRA, and w/o SG using a tiny backbone: architecture-compatible initialization from CP, expected newly initialized VRA tensors, finite forward outputs, and nested probabilities.
- Direct tensor checks passed for identical SG-on/off forward values, the intended direct disc-to-cup gradient switch, zero-initialized/bounded VRA correction, and finite soft-vCDR loss gradients.
- Copied-file source hashes and package checksums are verified by `python3 -m scripts.check_release`. The common scorer remains identical to its archived source. The probability-analysis wrapper adds preflight checks and probability-index export/reuse while retaining numerical formulas; original and adapted hashes are recorded separately.
- The complete pytest suite passed locally: **60 tests and 33 subtests**. This includes the archived geometry/coupling tests, protocol and release-hygiene tests, and probability-analysis tests.
- A synthetic analysis exports FP32 probabilities, then reuses them through the generated job specification. Per-image metrics, proxy statistics, and macro summaries match exactly; the original probability arrays and index remain unchanged. Empty predicted discs retain undefined hard-vCDR values and explicit validity counts. Modified arrays are rejected even after their index checksums are updated, including under `python -O`.
- The end-to-end CPU smoke test passed using eight synthetic images and a tiny backbone: one CP epoch, one Full epoch initialized from CP, image-only frozen prediction, and offline scoring. Inference loads no target labels, and hard-mask containment violations are zero.
- GitHub Actions runs 28 dependency-light checks and three dry-run commands on Python 3.10/3.12. A Python 3.11 CPU job runs the complete pytest suite and synthetic training/evaluation workflow. It downloads runtime dependencies, but no datasets or pretrained model weights, and uploads no experiment artifacts.
- The clean directory contains no real-data manifests, retinal images/masks, model weights, logs, remote server paths, or remote launchers. The SVGs under `assets/` are original vector diagrams and icons, not dataset samples. A basic source-pattern scan found no obvious embedded access tokens/private keys; this is not a comprehensive credential or license audit.

## Runtime boundary

Fresh local checks on 2026-09-27 used Python 3.11.15, PyTorch 2.7.1+cu118, torchvision 0.22.1+cu118, timm 1.0.28, NumPy 2.4.6, SciPy 1.17.1, and pytest 9.1.1, with CUDA disabled. An isolated project environment reused installed PyTorch and installed missing dependencies without modifying the existing environment. These versions meet `pyproject.toml`; the earlier checks in an older, incomplete environment are superseded by this run. GitHub Actions uses the corresponding CPU-only PyTorch/torchvision builds.

The tests verify execution and geometry on synthetic inputs. Full 240-epoch training, dataset converters on restricted datasets, the pretrained DINOv3 path, GPU execution, and paper-level numerical reproduction were not run in this validation. Full paper execution requires the separately obtained datasets and exact pretrained backbone described in [DATA.md](DATA.md).
