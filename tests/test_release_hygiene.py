"""Regression checks for the public source inventory boundary."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import check_release


class ReleaseHygieneTests(unittest.TestCase):
    def test_git_tracked_private_file_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "data" / "private.csv").write_text("patient_id\nexample\n")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "add", "-f", "data/private.csv"], check=True)
            with patch.object(check_release, "ROOT", root):
                self.assertEqual(check_release.private_release_paths(), [Path("data/private.csv")])

    def test_source_archive_private_file_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights").mkdir()
            (root / "weights" / "model.safetensors").write_bytes(b"example")
            with patch.object(check_release, "ROOT", root):
                self.assertEqual(check_release.private_release_paths(), [Path("weights/model.safetensors")])

    def test_checks_remain_active_under_optimized_python(self) -> None:
        result = subprocess.run(
            [sys.executable, "-O", "-c", "from scripts.check_release import require; require(False, 'blocked')"],
            cwd=check_release.ROOT, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("blocked", result.stderr)


if __name__ == "__main__":
    unittest.main()
