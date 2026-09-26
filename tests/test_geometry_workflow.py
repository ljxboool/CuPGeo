"""Dependency-free checks for geometry job and probability-file bookkeeping."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts import analyze_ratio_geometry as geometry


DOMAINS = ["binrushed", "magrabia", "rim_one", "papila"]


def complete_spec() -> dict:
    return {
        "target_domains": DOMAINS.copy(),
        "jobs": [
            {
                "method": method,
                "seed": 0,
                "domain": domain,
                "artifact": f"runs/{method}/{domain}.pt",
                "manifest": f"manifests/{domain}.csv",
                "data_root": "data/fundus",
            }
            for method in ("full", "no_ratio")
            for domain in DOMAINS
        ],
        "pairs": [{"before": "no_ratio", "after": "full"}],
    }


class GeometrySpecTests(unittest.TestCase):
    def test_complete_matrix_and_reuse_requirement(self) -> None:
        spec = complete_spec()
        geometry.validate_spec(spec)
        with self.assertRaises(ValueError):
            geometry.validate_spec(spec, reuse_probabilities=True)
        for job in spec["jobs"]:
            job["probability_index"] = (
                f"runs/{job['method']}/{job['domain']}/probabilities/index.json"
            )
        geometry.validate_spec(spec, reuse_probabilities=True)

    def test_reject_malformed_and_duplicate_jobs(self) -> None:
        cases = [None, {}, {"jobs": []}, {"jobs": "invalid"}]
        missing_artifact = complete_spec()
        del missing_artifact["jobs"][0]["artifact"]
        cases.append(missing_artifact)
        duplicate = complete_spec()
        duplicate["jobs"].append(copy.deepcopy(duplicate["jobs"][0]))
        cases.append(duplicate)
        for spec in cases:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                geometry.validate_spec(spec)

    def test_reject_unsafe_job_names_and_noninteger_seeds(self) -> None:
        cases = [
            ("method", "../full"),
            ("method", "/full"),
            ("method", "-full"),
            ("method", ""),
            ("domain", "../binrushed"),
            ("domain", "bin rushed"),
            ("seed", -1),
            ("seed", True),
            ("seed", 0.5),
            ("seed", "0"),
        ]
        for field, value in cases:
            spec = complete_spec()
            spec["jobs"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                geometry.validate_spec(spec)

    def test_reject_incomplete_domains_and_unmatched_pairs(self) -> None:
        missing_domain = complete_spec()
        missing_domain["jobs"].pop()
        repeated_domain = complete_spec()
        repeated_domain["target_domains"][-1] = DOMAINS[0]
        unknown_pair = complete_spec()
        unknown_pair["pairs"][0]["after"] = "unknown"
        unmatched_seed = complete_spec()
        for job in copy.deepcopy(unmatched_seed["jobs"][:len(DOMAINS)]):
            job["seed"] = 1
            unmatched_seed["jobs"].append(job)
        for spec in (missing_domain, repeated_domain, unknown_pair, unmatched_seed):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                geometry.validate_spec(spec)


class GeometryProbabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.private_root = self.root / "runs"
        self.index_path = self.private_root / "analysis" / "probabilities" / "index.json"
        self.index_path.parent.mkdir(parents=True)
        self.artifact = self.private_root / "predictions.pt"
        self.artifact.write_bytes(b"synthetic artifact bytes for hash verification")
        self.image_ids = ["image-a", "image-b"]
        self.records = []
        for position, image_id in enumerate(self.image_ids):
            path = self.index_path.parent / f"{position:04d}.npy"
            # No numerical loading occurs in these bookkeeping tests.
            payload = f"synthetic probability bytes {position}".encode()
            path.write_bytes(payload)
            self.records.append({
                "image_id": image_id,
                "file": path.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "shape": [2, 768, 768],
            })
        geometry.export_probability_index(self.index_path, self.artifact, self.records)

    def test_exported_index_is_reusable_without_rewriting_probabilities(self) -> None:
        paths = [self.index_path.parent / record["file"] for record in self.records]
        before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
        index = geometry.load_probability_index(self.index_path, self.artifact, self.image_ids)
        self.assertEqual(index["channels"], ["OD", "OC"])
        self.assertEqual(index["probability_dtype"], "float32")
        self.assertEqual(index["source_sha256"], hashlib.sha256(self.artifact.read_bytes()).hexdigest())
        self.assertEqual(index["records"], self.records)
        for record, expected in zip(index["records"], paths):
            self.assertEqual(geometry.verified_probability_path(self.index_path, record), expected.resolve())
        self.assertEqual(before, [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths])

    def test_reject_changed_source_artifact(self) -> None:
        self.artifact.write_bytes(b"changed source artifact")
        with self.assertRaises(ValueError):
            geometry.load_probability_index(self.index_path, self.artifact, self.image_ids)

    def test_reject_changed_index_metadata_and_image_order(self) -> None:
        original = json.loads(self.index_path.read_text())
        changes = [
            ("channels", ["OC", "OD"]),
            ("probability_dtype", "float16"),
            ("source_sha256", "0" * 64),
            ("records", list(reversed(self.records))),
            ("records", self.records[:1]),
        ]
        for field, value in changes:
            index = copy.deepcopy(original)
            index[field] = value
            self.index_path.write_text(json.dumps(index))
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                geometry.load_probability_index(self.index_path, self.artifact, self.image_ids)

    def test_reject_changed_probability_hash_and_shape(self) -> None:
        record = copy.deepcopy(self.records[0])
        record["shape"] = [2, 384, 384]
        with self.assertRaises(ValueError):
            geometry.verified_probability_path(self.index_path, record)
        record = copy.deepcopy(self.records[0])
        record["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            geometry.verified_probability_path(self.index_path, record)
        path = self.index_path.parent / self.records[0]["file"]
        path.write_bytes(b"changed probability bytes")
        with self.assertRaises(ValueError):
            geometry.verified_probability_path(self.index_path, self.records[0])

    def test_reject_probability_traversal_absolute_paths_and_symlink_escape(self) -> None:
        outside = self.private_root / "outside.npy"
        outside.write_bytes((self.index_path.parent / self.records[0]["file"]).read_bytes())
        link = self.index_path.parent / "linked.npy"
        link.symlink_to(outside)
        inside = self.index_path.parent / self.records[0]["file"]
        for filename in ("../../outside.npy", str(outside), str(inside), link.name):
            record = dict(self.records[0], file=filename)
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                geometry.verified_probability_path(self.index_path, record)

    def test_private_output_requires_resolved_child_of_private_root(self) -> None:
        output = self.private_root / "new" / "analysis"
        self.assertEqual(geometry.private_output(output, self.private_root), output.resolve())
        outside = self.root / "public"
        outside.mkdir()
        link = self.private_root / "linked-output"
        link.symlink_to(outside, target_is_directory=True)
        for path in (self.private_root, outside, self.private_root / ".." / "public", link / "analysis"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                geometry.private_output(path, self.private_root)
        self.assertFalse(output.exists())

    def test_integrity_checks_remain_active_under_optimized_python(self) -> None:
        self.artifact.write_bytes(b"changed source artifact")
        code = (
            "import sys\n"
            "from pathlib import Path\n"
            "from scripts.analyze_ratio_geometry import load_probability_index\n"
            "load_probability_index(Path(sys.argv[1]), Path(sys.argv[2]), ['image-a', 'image-b'])\n"
        )
        result = subprocess.run(
            [sys.executable, "-O", "-c", code, str(self.index_path), str(self.artifact)],
            cwd=Path(geometry.__file__).resolve().parents[1], capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ValueError", result.stderr)


if __name__ == "__main__":
    unittest.main()
