import unittest

import numpy as np

from edge_finer_research.evaluation.rethreshold import (
    measure_thresholds,
    reference_masks,
    threshold_mask,
)
from edge_finer_research.evaluation.thresholds import select_pixel_threshold


class RethresholdTests(unittest.TestCase):
    def test_zero_cam_stays_empty_at_small_positive_threshold(self):
        prediction = threshold_mask(np.zeros((3, 3), dtype=np.float32), 0.001)
        self.assertFalse(prediction.any())

    def test_measures_area_and_selectable_metrics(self):
        cam = np.array([[0.9, 0.2], [0.0, 0.0]], dtype=np.float32)
        mask = np.array([[1, 0], [0, 0]], dtype=bool)
        scores = measure_thresholds([cam], [mask], [0.1, 0.5])
        self.assertEqual(scores[1].pixel_f1, 1.0)
        self.assertEqual(scores[1].iou, 1.0)
        self.assertEqual(scores[1].predicted_area_ratio, 0.25)
        self.assertEqual(scores[1].ground_truth_area_ratio, 0.25)

    def test_selection_order_matches_existing_threshold_protocol(self):
        cam = np.array([[0.9, 0.2], [0.0, 0.0]], dtype=np.float32)
        mask = np.array([[1, 0], [0, 0]], dtype=bool)
        scores = measure_thresholds([cam], [mask], [0.1, 0.5])
        new_best = max(scores, key=lambda item: (item.pixel_f1, item.iou, -item.threshold))
        existing = select_pixel_threshold([cam], [mask], [0.1, 0.5])
        self.assertEqual(new_best.threshold, existing.threshold)
        self.assertEqual(new_best.pixel_f1, existing.pixel_f1)
        self.assertEqual(new_best.iou, existing.iou)

    def test_empty_and_full_references(self):
        mask = np.array([[1, 0], [0, 0]], dtype=bool)
        references = reference_masks([mask])
        self.assertEqual(references["empty"]["predicted_area_ratio"], 0.0)
        self.assertEqual(references["full_image"]["predicted_area_ratio"], 1.0)
        self.assertEqual(references["full_image"]["iou"], 0.25)

    def test_rejects_non_float_or_nonfinite_cam(self):
        with self.assertRaises(ValueError):
            threshold_mask(np.zeros((2, 2), dtype=np.uint8), 0.1)
        with self.assertRaises(ValueError):
            threshold_mask(np.array([[np.nan]], dtype=np.float32), 0.1)


if __name__ == "__main__":
    unittest.main()
