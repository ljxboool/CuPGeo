# Packaging validation

This bundle is a source release, not a newly reproduced experiment. Publishing it does not run full training or verify the paper's numerical results.

The manuscript currently prints the same CuPGeo macro values in the main comparison and the matched ablation table, while its protocol distinguishes a 120-epoch single-stage main run from a 120+120-epoch ablation run. This release keeps those workflows separate. The numerical rows should be tied to their original checkpoint hashes before claiming that each table has been independently reproduced.

## Checks performed

- Python AST syntax checking: all packaged Python files parsed successfully.
- Configuration inheritance: all 14 YAML configurations resolved. One stage is 120 epochs, inputs are 768 square, and data/weight locations are relative paths.
- Ablation isolation: w/o SG differs only in `model.detach_cup_from_od`; w/o VRA changes only the geometry head, calibration switch, allocation switch, and allocation weight. Ratio weights and remaining shared objectives are preserved.
- Dependency-free protocol tests cover common per-seed CP initialization, single-stage separation, CP reuse for the ratio sweep, the four-target evaluation matrix, and domain-before-seed aggregation.
- Dry-run command generation passed: matched seeds 0/1/2 produce 18 training commands (3 initializers and 15 stage-2 jobs); the default single-stage set produces 9. A six-weight ratio sweep with CP reuse produces 18 further stage-2 commands. The four-target full-model evaluation plan produces 12 prediction/scoring pairs. No training or real-data evaluation commands were executed by these checks.
- Focused CPU checks passed for CP, full, w/o ratio, w/o VRA, and w/o SG using a tiny backbone: architecture-compatible initialization from CP, expected newly initialized VRA tensors, finite forward outputs, and nested probabilities.
- Direct tensor checks passed for identical SG-on/off forward values, the intended direct disc-to-cup gradient switch, zero-initialized/bounded VRA correction, and finite soft-vCDR loss gradients.
- Copied-file source hashes and package checksums are verified by `python -m scripts.check_release`. The common scorer and probability-analysis files are identical to the selected archived sources.
- The clean directory contains no real-data manifests, retinal images/masks, model weights, logs, remote server paths, or remote launchers. The SVGs under `assets/` are original vector diagrams and icons, not dataset samples. A basic source-pattern scan found no obvious embedded access tokens/private keys; this is not a comprehensive credential or license audit.

## Runtime boundary

The archived geometry tests and synthetic smoke workflow are included. A complete environment meeting `pyproject.toml`, separately obtained datasets, and pretrained weights are required for full paper execution. Historical validation records in the parent package are not presented as fresh validation of this release.

The local focused CPU checks used Python 3.10.20 / PyTorch 2.0.0+cu118 with CUDA disabled. This environment is older than the declared project requirements and is **not** an approved reproduction environment. An attempt to collect the archived hierarchy/coupling tests stopped because OpenCV (`cv2`) was missing. The full pytest suite, dataset converters, DINOv3 path, end-to-end training smoke test, and real-data runs were therefore not validated here. Missing packages and the environment version are recorded rather than silently weakening the project's requirements or modifying the user's installed environment.

No optimizer steps were executed in the focused checks. No dependency downloads, new numerical paper results, or GPU training were performed.
