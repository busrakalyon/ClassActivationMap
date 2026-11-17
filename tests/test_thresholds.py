import unittest

import numpy as np

from edge_finer_research.evaluation.thresholds import select_pixel_threshold


class ThresholdSelectionTests(unittest.TestCase):
    def test_selects_using_only_provided_validation_examples(self):
        heatmap = np.array([[0.9, 0.6], [0.4, 0.1]])
        mask = np.array([[1, 0], [0, 0]], dtype=bool)
        result = select_pixel_threshold([heatmap], [mask], [0.3, 0.7])
        self.assertEqual(result.threshold, 0.7)
        self.assertEqual(result.pixel_f1, 1.0)

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            select_pixel_threshold([], [], [0.5])


if __name__ == "__main__":
    unittest.main()
