"""Exercise CP -> CuPGeo -> frozen predictions -> scoring with synthetic data."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/smoke"))
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Use a new or empty output directory; existing runs are preserved.")
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")

    from c3tta.data.synthetic import make_synthetic_manifest

    source = out / "source_data"
    manifest, _ = make_synthetic_manifest(source, count=8, image_size=64)
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for split in ("train", "val"):
        with (source / f"{split}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(row for row in rows if row["split"] == split)
    target = out / "target_data"
    target_eval, target_images = make_synthetic_manifest(target, count=8, image_size=64)
    root = Path(__file__).resolve().parents[1]

    def run(module: str, *args: object) -> None:
        subprocess.run([sys.executable, "-m", module, *map(str, args)], check=True, env=env)

    for name in ("cp_baseline", "cupgeo"):
        cfg = yaml.safe_load((root / "configs" / f"{name}.yaml").read_text())
        cfg["data"].update(root=str(source), train_manifest=str(source / "train.csv"),
                           val_manifest=str(source / "val.csv"))
        cfg["model"].update(backbone="tiny", checkpoint=None, image_size=64,
                            feature_indices=[0, 1, 2, 3], decoder_channels=16,
                            adapter_rank=4, classifier_hidden_dim=16,
                            lora_rank=0, freeze_backbone=False)
        cfg["train"].update(epochs=1, batch_size=2, grad_accum_steps=1,
                            num_workers=0, precision="fp32", lr_warmup_epochs=0)
        path = out / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        extra = ["--init-checkpoint", out / "cp_baseline/best.pt"] if name == "cupgeo" else []
        run("scripts.train_source", "--config", path, "--output", out / name, *extra)

    artifact = out / "target.predictions.pt"
    run("scripts.evaluate", "--config", out / "cupgeo.yaml",
        "--checkpoint", out / "cupgeo/best.pt", "--csv", target_images,
        "--data-root", target, "--predictions", artifact, "--artifact-method", "CuPGeo")
    run("scripts.score_predictions", "--prediction", f"CuPGeo={artifact}",
        "--eval-csv", target_eval, "--data-root", target,
        "--private-root", out, "--output-dir", out / "scores", "--bootstrap-samples", 0)
    summary = json.loads((out / "scores/summary.json").read_text())
    method = summary["methods"]["CuPGeo"]
    if method["metrics"]["topology_violation"] != 0:
        raise RuntimeError("The nested model produced a containment violation.")
    if method["artifact_metadata"]["inference_target_labels_loaded"] is not False:
        raise RuntimeError("Prediction did not use the image-only path.")
    print(f"Smoke test passed: two training stages, prediction, scoring; outputs in {out}")
    print("Synthetic examples exercise execution only; scores are not paper results.")


if __name__ == "__main__":
    main()
