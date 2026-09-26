"""CPU-only export/reuse integration checks using entirely synthetic inputs."""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class GeometryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dependencies = {}
        for name in ("numpy", "torch", "scipy", "cv2", "PIL.Image", "sklearn"):
            try:
                dependencies[name] = importlib.import_module(name)
            except (ImportError, OSError) as error:
                raise unittest.SkipTest(f"Geometry integration requires {name}: {error}") from error
        cls.np = dependencies["numpy"]
        cls.torch = dependencies["torch"]
        cls.Image = dependencies["PIL.Image"]

    def run_analysis(self, spec: Path, output: Path, private_root: Path,
                     *, reuse: bool = False, optimized: bool = False) -> subprocess.CompletedProcess:
        command = [sys.executable]
        if optimized:
            command.append("-O")
        command.extend([
            "-m", "scripts.analyze_ratio_geometry",
            "--spec", str(spec), "--output", str(output),
            "--source-root", str(ROOT), "--private-root", str(private_root),
            "--threads", "1", "--compute-proxy",
        ])
        if reuse:
            command.append("--reuse-probabilities")
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[name] = "1"
        return subprocess.run(
            command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60,
        )

    def create_inputs(self, private_root: Path) -> Path:
        np, torch = self.np, self.torch
        data_root = private_root / "synthetic_data"
        data_root.mkdir(parents=True)
        image_ids = ["synthetic_valid", "synthetic_empty_disc"]
        rows = []
        for index, image_id in enumerate(image_ids):
            disc = np.zeros((24, 16), dtype=np.uint8)
            cup = np.zeros_like(disc)
            if index == 0:
                disc[4:20, 3:13] = 255
                cup[8:16, 5:11] = 255
            else:
                disc[2:22, 3:13] = 255
                cup[10:16, 5:11] = 255
            image_name = f"{image_id}.png"
            disc_name, cup_name = f"{image_id}_od.png", f"{image_id}_oc.png"
            self.Image.new("RGB", (16, 24)).save(data_root / image_name)
            self.Image.fromarray(disc).save(data_root / disc_name)
            self.Image.fromarray(cup).save(data_root / cup_name)
            rows.append({
                "image": image_name, "image_id": image_id,
                "patient_id": f"synthetic_patient_{index}", "device": "synthetic",
                "split": "test", "od_mask": disc_name, "oc_mask": cup_name,
            })
        manifest = private_root / "synthetic_manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        logits = torch.full((2, 2, 768, 768), -4.0, dtype=torch.float16, device="cpu")
        logits[0, 0, 128:640, 144:624] = 4.0
        logits[0, 1, 256:512, 240:528] = 4.0
        artifact = private_root / "synthetic_predictions.pt"
        torch.save({
            "schema_version": "c3tta.predictions.v1",
            "metadata": {
                "source_seed": 0,
                "checkpoint_sha256": hashlib.sha256(b"synthetic checkpoint identity").hexdigest(),
                "checkpoint_epoch": 1,
                "inference_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            },
            "image_ids": image_ids,
            "predictions": {
                "seg_logits": logits,
                "soft_vcdr": torch.tensor([0.5, 1.0], dtype=torch.float32, device="cpu"),
            },
        }, artifact)
        spec = private_root / "geometry_jobs.json"
        spec.write_text(json.dumps({
            "target_domains": ["binrushed"],
            "jobs": [{
                "method": "full", "seed": 0, "domain": "binrushed",
                "artifact": str(artifact), "manifest": str(manifest), "data_root": str(data_root),
            }],
            "pairs": [],
        }), encoding="utf-8")
        return spec

    def test_export_reuse_and_reject_rehashed_array_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            private_root = Path(directory) / "runs"
            spec = self.create_inputs(private_root)
            export_output, reuse_output = private_root / "export", private_root / "reuse"
            result = self.run_analysis(spec, export_output, private_root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            job_relative = Path("full") / "seed0" / "binrushed"
            index_path = export_output / job_relative / "probabilities" / "index.json"
            index = json.loads(index_path.read_text())
            self.assertEqual([record["image_id"] for record in index["records"]],
                             ["synthetic_valid", "synthetic_empty_disc"])
            probability_paths = [index_path.parent / record["file"] for record in index["records"]]
            original_files = [index_path, *probability_paths]
            before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in original_files}

            reuse_spec = export_output / "reuse_jobs.json"
            result = self.run_analysis(reuse_spec, reuse_output, private_root, reuse=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for relative in (Path("per_image.csv"), job_relative / "per_image.csv"):
                with (export_output / relative).open(newline="") as stream:
                    exported_rows = list(csv.DictReader(stream))
                with (reuse_output / relative).open(newline="") as stream:
                    reused_rows = list(csv.DictReader(stream))
                self.assertEqual(exported_rows, reused_rows)
                self.assertEqual(len(exported_rows), 2)
                self.assertEqual(exported_rows[0]["hard_vcdr_valid"], "1")
                self.assertEqual(float(exported_rows[0]["hard_vcdr_absolute_error"]), 0.0)
                self.assertEqual(exported_rows[1]["hard_vcdr_valid"], "0")
                for field in ("hard_vcdr_prediction", "hard_vcdr_signed_error", "hard_vcdr_absolute_error"):
                    self.assertEqual(exported_rows[1][field], "")

            exported_summary = json.loads((export_output / "summary.json").read_text())
            reused_summary = json.loads((reuse_output / "summary.json").read_text())
            self.assertEqual(exported_summary["aggregates"], reused_summary["aggregates"])
            for summary in (exported_summary, reused_summary):
                self.assertFalse(summary["cuda_used"])
                self.assertEqual(summary["n_images"], 2)
                self.assertEqual(summary["n_probability_files"], 2)
            exported_job = json.loads((export_output / job_relative / "summary.json").read_text())
            reused_job = json.loads((reuse_output / job_relative / "summary.json").read_text())
            self.assertFalse(exported_job["probability_files_reused"])
            self.assertTrue(reused_job["probability_files_reused"])
            self.assertEqual(exported_job["metrics"], reused_job["metrics"])
            self.assertEqual(exported_job["proxy_statistics"], reused_job["proxy_statistics"])
            self.assertEqual(exported_job["n_vcdr_valid"], 1)
            pair_statistics = exported_job["proxy_statistics"]["soft_prediction_vs_hard_prediction"]
            self.assertEqual(pair_statistics["n"], 1)
            self.assertEqual(pair_statistics["invalid"], 1)
            self.assertIsNone(pair_statistics["pearson"])
            self.assertIsNone(pair_statistics["spearman"])
            self.assertEqual(before, {
                path: (path.read_bytes(), path.stat().st_mtime_ns) for path in original_files
            })

            # A new valid file checksum must not bypass equality with frozen logits.
            array = self.np.load(probability_paths[0], allow_pickle=False)
            self.assertEqual(array.dtype, self.np.float32)
            self.assertEqual(array.shape, (2, 768, 768))
            array[0, 0, 0] = self.np.float32(0.25)
            self.np.save(probability_paths[0], array, allow_pickle=False)
            index["records"][0]["sha256"] = hashlib.sha256(probability_paths[0].read_bytes()).hexdigest()
            index_path.write_text(json.dumps(index), encoding="utf-8")
            rejected_output = private_root / "rejected"
            result = self.run_analysis(
                reuse_spec, rejected_output, private_root, reuse=True, optimized=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Saved probabilities differ from artifact", result.stderr)
            self.assertFalse((rejected_output / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
