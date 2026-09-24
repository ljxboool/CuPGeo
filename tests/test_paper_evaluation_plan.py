"""Dependency-free checks of the paper evaluation command matrix."""
import unittest
from pathlib import Path

from scripts.plan_paper_evaluation import build_plan


class PaperEvaluationPlanTests(unittest.TestCase):
    def test_complete_four_target_matrix_and_source_only_order(self):
        jobs = build_plan("matched", [0, 1, 2], ["full", "no_vra"], Path("runs/paper"))
        self.assertEqual(len(jobs), 24)
        self.assertEqual({job.manifest.name for job in jobs},
                         {"binrushed_test.csv", "magrabia_test.csv", "rim_one_eval.csv", "papila_eval.csv"})
        self.assertTrue(all("--predictions" in job.predict_command for job in jobs))
        self.assertTrue(all("--segmentation-threshold" in job.score_command for job in jobs))
        self.assertTrue(all("--tta-horizontal-flip" not in job.predict_command for job in jobs))
        self.assertTrue(all("stage2_" in str(job.checkpoint) for job in jobs))

    def test_single_stage_has_no_matched_checkpoint(self):
        jobs = build_plan("single", [0], ["full", "mixstyle", "dsu"], Path("runs/paper"))
        self.assertEqual(len(jobs), 12)
        self.assertTrue(all("stage2_" not in str(job.checkpoint) for job in jobs))

    def test_reject_wrong_suite_variant(self):
        with self.assertRaises(ValueError):
            build_plan("single", [0], ["no_sg"], Path("runs/paper"))


if __name__ == "__main__":
    unittest.main()
