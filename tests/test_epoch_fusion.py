import unittest

import numpy as np

from edge_finer_research.cams.epoch_fusion import fuse_epoch_cams


class EpochFusionTests(unittest.TestCase):
    def test_max_ignores_invalid_epochs(self):
        result = fuse_epoch_cams(
            [np.full((2, 2), 0.2), np.full((2, 2), 0.9)], valid=[True, False], method="max"
        )
        np.testing.assert_allclose(result.map, 0.2)
        self.assertEqual(result.valid_count, 1)

    def test_no_fake_prediction_returns_zero_map(self):
        result = fuse_epoch_cams([np.ones((2, 2))], valid=[False])
        np.testing.assert_array_equal(result.map, 0.0)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            fuse_epoch_cams([np.zeros((2, 2)), np.zeros((3, 3))])


if __name__ == "__main__":
    unittest.main()
