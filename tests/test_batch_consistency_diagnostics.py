import importlib.util
import unittest

import numpy as np

from edge_finer_research.diagnostics.batch_consistency import (
    CUDA_PROFILES,
    mask_iou,
    numerical_comparison,
    pearson,
)


class BatchConsistencyMetricTests(unittest.TestCase):
    def test_comparison_reports_mean_max_correlation_without_relaxing_tolerance(self):
        left = np.array([0.0, 0.5, 1.0])
        right = np.array([0.0, 0.5001, 0.9])
        result = numerical_comparison(left, right, atol=1e-5, rtol=1e-4)
        self.assertFalse(result["allclose"])
        self.assertAlmostEqual(result["max_abs_difference"], 0.1)
        self.assertGreater(result["pearson"], 0.99)

    def test_mask_iou_distinguishes_threshold_instability(self):
        left = np.array([[0.1, 0.9], [0.2, 0.8]])
        right = np.array([[0.1, 0.4], [0.2, 0.8]])
        self.assertAlmostEqual(mask_iou(left, right, 0.5), 0.5)

    def test_constant_equal_arrays_have_unit_correlation(self):
        values = np.zeros((2, 2))
        self.assertEqual(pearson(values, values), 1.0)

    def test_backend_matrix_contains_all_requested_combinations(self):
        combinations = {(item["tf32"], item["deterministic"]) for item in CUDA_PROFILES.values()}
        self.assertEqual(combinations, {(True, True), (True, False), (False, True), (False, False)})


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed locally")
class ModelModeAuditTests(unittest.TestCase):
    def test_eval_audit_detects_batchnorm_and_dropout_state(self):
        import torch

        from edge_finer_research.diagnostics.batch_consistency import model_eval_audit

        model = torch.nn.Sequential(
            torch.nn.BatchNorm2d(2),
            torch.nn.Dropout(0.5),
        ).eval()
        report = model_eval_audit(model)
        self.assertTrue(report["all_batch_norm_eval"])
        self.assertTrue(report["all_dropout_eval"])
        self.assertEqual(report["training_modules"], [])


if __name__ == "__main__":
    unittest.main()
