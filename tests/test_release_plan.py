"""Dependency-free tests: python -m unittest discover -s tests -p test_release_plan.py."""
import unittest
from pathlib import Path

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

    def test_single_stage_never_uses_cp_initialization(self):
        jobs = build_plan("single", [0, 1, 2], ["full", "mixstyle", "dsu"], Path("runs/paper"))
        self.assertEqual(len(jobs), 9)
        self.assertTrue(all("--init-checkpoint" not in job.command for job in jobs))

    def test_reject_duplicate_seeds(self):
        with self.assertRaises(ValueError):
            build_plan("matched", [0, 0], ["full"], Path("runs"))

    def test_reject_sg_as_single_stage_paper_protocol(self):
        with self.assertRaises(ValueError):
            build_plan("single", [0], ["no_sg"], Path("runs"))

    def test_reject_duplicate_variants(self):
        with self.assertRaises(ValueError):
            build_plan("matched", [0], ["full", "full"], Path("runs"))


if __name__ == "__main__":
    unittest.main()
