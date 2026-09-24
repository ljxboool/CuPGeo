"""Dependency-free tests: python -m unittest discover -s tests -p test_release_plan.py."""
import unittest
import hashlib
from pathlib import Path
import tempfile
from unittest.mock import patch

from scripts import plan_experiments
from scripts.plan_experiments import build_plan


class ReleasePlanTests(unittest.TestCase):
    def test_all_stage2_variants_share_one_cp_initializer_per_seed(self):
        jobs = build_plan("matched", [0, 1, 2], ["cp", "full", "no_ratio", "no_vra", "no_sg"], Path("runs/paper"))
        self.assertEqual(len(jobs), 18)
        for seed in range(3):
            group = jobs[seed * 6:(seed + 1) * 6]
            self.assertNotIn("--init-checkpoint", group[0].command)
            for job in group[1:]:
                index = job.command.index("--init-checkpoint")
                self.assertEqual(job.command[index + 1], str(group[0].output / "best.pt"))
                self.assertEqual(job.command[job.command.index("--seed") + 1], str(seed))
                self.assertEqual(job.command[job.command.index("--epochs") + 1], "120")

    def test_main_baselines_use_240_epochs_without_cp_initialization(self):
        jobs = build_plan("single", [0, 1, 2], ["mixstyle", "dsu"], Path("runs/paper"))
        self.assertEqual(len(jobs), 6)
        self.assertTrue(all("--init-checkpoint" not in job.command for job in jobs))
        self.assertTrue(all(job.command[job.command.index("--epochs") + 1] == "240" for job in jobs))

    def test_reject_duplicate_seeds(self):
        with self.assertRaises(ValueError):
            build_plan("matched", [0, 0], ["full"], Path("runs"))

    def test_reject_sg_as_single_stage_paper_protocol(self):
        with self.assertRaises(ValueError):
            build_plan("single", [0], ["no_sg"], Path("runs"))

    def test_reject_single_stage_full_as_table1_checkpoint(self):
        with self.assertRaises(ValueError):
            build_plan("single", [0], ["full"], Path("runs"))

    def test_reject_duplicate_variants(self):
        with self.assertRaises(ValueError):
            build_plan("matched", [0], ["full", "full"], Path("runs"))

    def test_ratio_sweep_uses_matched_cp_per_seed(self):
        variants = ["ratio_025", "ratio_050", "ratio_100", "ratio_150", "full", "ratio_250", "ratio_300"]
        jobs = build_plan("matched", [0, 1, 2], variants, Path("runs/paper/ratio_study"))
        self.assertEqual(len(jobs), 24)
        for seed in range(3):
            group = jobs[seed * 8:(seed + 1) * 8]
            cp = str(group[0].output / "best.pt")
            for job in group[1:]:
                self.assertEqual(job.command[job.command.index("--init-checkpoint") + 1], cp)
                self.assertEqual(job.command[job.command.index("--epochs") + 1], "120")
        self.assertIn("cupgeo.yaml", jobs[5].command[jobs[5].command.index("--config") + 1])

    def test_ratio_sweep_can_reuse_existing_cp(self):
        jobs = build_plan("matched", [0, 1, 2], ["ratio_025", "ratio_050"],
                          Path("runs/paper"), reuse_cp=True)
        self.assertEqual(len(jobs), 6)
        self.assertEqual(jobs[0].command[jobs[0].command.index("--init-checkpoint") + 1],
                         "runs/paper/matched/seed0/stage1_cp/best.pt")
        with self.assertRaises(ValueError):
            build_plan("single", [0], ["full"], Path("runs/paper"), reuse_cp=True)

    def test_paper_inputs_require_source_split_and_original_backbone(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data").mkdir()
            train = root / "train.csv"
            val = root / "val.csv"
            train.write_text("image_id,source_domain\n" + "".join(f"tr{i},REFUGE\n" for i in range(320)))
            val.write_text("image_id,source_domain\n" + "".join(f"va{i},REFUGE\n" for i in range(80)))
            weight = root / "model.safetensors"
            weight.write_bytes(b"test-backbone")
            paths = {"data-root": str(root / "data"), "train-manifest": str(train),
                     "val-manifest": str(val), "backbone-checkpoint": str(weight)}
            digest = hashlib.sha256(weight.read_bytes()).hexdigest()
            with patch.object(plan_experiments, "PAPER_BACKBONE_SHA256", digest):
                plan_experiments.verify_paper_inputs(paths)
                val.write_text("image_id,source_domain\ntr0,REFUGE\n" +
                               "".join(f"va{i},REFUGE\n" for i in range(79)))
                with self.assertRaisesRegex(ValueError, "overlap"):
                    plan_experiments.verify_paper_inputs(paths)


if __name__ == "__main__":
    unittest.main()
