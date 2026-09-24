"""Dependency-free tests for the four-domain paper aggregation protocol."""
import unittest

from scripts.aggregate_paper_domains import DOMAINS, aggregate


def sample_summaries():
    summaries = {}
    for seed in (0, 1, 2):
        for index, (domain, (filename, count)) in enumerate(DOMAINS.items()):
            offset = seed * 0.02 + index * 0.01
            od, oc = 0.8 + offset, 0.6 + offset
            summaries[(seed, domain)] = {
                "schema_version": "c3tta.offline_scores.v1",
                "evaluation_manifest": {"filename": filename, "num_samples": count,
                                        "sha256": f"{index + 1:064x}"},
                "segmentation_thresholds": {"od": 0.5, "oc": 0.5},
                "methods": {"CuPGeo": {
                    "artifact_metadata": {
                        "method": "CuPGeo", "source_seed": seed,
                        "config_filename": "cupgeo.yaml",
                        "checkpoint_selection_metric": "val_seg_dice",
                        "inference_precision": "fp16",
                        "optimization_applied": False,
                        "inference_target_labels_loaded": False,
                        "inference_tta_image_sizes": [768],
                        "inference_tta_horizontal_flip": False,
                        "checkpoint_sha256": f"{seed + 1:064x}",
                    },
                    "metrics": {"od_dice": od, "oc_dice": oc, "dice_mean": (od + oc) / 2,
                                "vcdr_mae": 0.1 + offset, "topology_violation": 0.0,
                                "vcdr_valid_samples": float(count),
                                "vcdr_invalid_predictions": 0.0},
                }},
            }
    return summaries


class PaperAggregationTests(unittest.TestCase):
    def test_domain_equal_weight_precedes_sample_sd(self):
        summaries = sample_summaries()
        summaries[(0, "binrushed")]["methods"]["CuPGeo"]["metrics"]["topology_violation"] = 0.1
        result = aggregate(summaries, seeds=(0, 1, 2), method="CuPGeo", variant="full")
        self.assertAlmostEqual(result["per_seed"]["0"]["macro4"]["dice_mean"], 0.715)
        self.assertAlmostEqual(result["summary"]["macro4"]["dice_mean"]["mean"], 0.735)
        self.assertAlmostEqual(result["summary"]["macro4"]["dice_mean"]["sample_sd"], 0.02)
        self.assertEqual(result["per_seed"]["0"]["domains"]["binrushed"]["vcdr_total_samples"], 39)
        self.assertAlmostEqual(result["per_seed"]["0"]["macro4"]["cvr_percent"], 2.5)

    def test_reject_incomplete_and_changed_protocol(self):
        summaries = sample_summaries()
        summaries.pop((2, "papila"))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            aggregate(summaries, seeds=(0, 1, 2), method="CuPGeo", variant="full")
        summaries = sample_summaries()
        summaries[(1, "papila")]["segmentation_thresholds"]["oc"] = 0.6
        with self.assertRaisesRegex(ValueError, "threshold"):
            aggregate(summaries, seeds=(0, 1, 2), method="CuPGeo", variant="full")


if __name__ == "__main__":
    unittest.main()
