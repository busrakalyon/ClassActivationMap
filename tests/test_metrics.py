import math
import unittest

import numpy as np

from edge_finer_research.evaluation.metrics import binary_auc, combined_f1, image_metrics, pixel_metrics


class MetricTests(unittest.TestCase):
    def test_paper_image_f1_is_sensitivity_specificity_harmonic(self):
        result = image_metrics([1, 1, 0, 0], [1, 0, 0, 0], [0.9, 0.4, 0.2, 0.1])
        self.assertAlmostEqual(result.sensitivity, 0.5)
        self.assertAlmostEqual(result.specificity, 1.0)
        self.assertAlmostEqual(result.paper_f1, 2.0 / 3.0)

    def test_auc_handles_ties(self):
        self.assertAlmostEqual(binary_auc([0, 1], [0.5, 0.5]), 0.5)

    def test_pixel_metrics_and_combined_f1(self):
        truth = np.array([[1, 1], [0, 0]], dtype=bool)
        prediction = np.array([[1, 0], [1, 0]], dtype=bool)
        result = pixel_metrics(truth, prediction)
        self.assertAlmostEqual(result.f1, 0.5)
        self.assertAlmostEqual(result.iou, 1.0 / 3.0)
        self.assertAlmostEqual(combined_f1(0.5, 0.5), 0.5)

    def test_single_class_auc_is_undefined(self):
        self.assertTrue(math.isnan(binary_auc([1, 1], [0.2, 0.8])))


if __name__ == "__main__":
    unittest.main()
