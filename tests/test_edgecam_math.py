import importlib.util
import unittest

import numpy as np

from edge_finer_research.cams.math_reference import paper_edgecam_numpy


class EdgeCAMMathTests(unittest.TestCase):
    def test_equations_two_to_four_on_constant_tensor(self):
        activation = np.ones((1, 1, 2, 2), dtype=np.float64)
        gradient = np.ones_like(activation)
        result = paper_edgecam_numpy(activation, gradient, normalize=False)
        np.testing.assert_allclose(result, np.full((1, 2, 2), 1.0 / 6.0))

    def test_negative_gradient_is_suppressed(self):
        activation = np.ones((1, 2, 2, 2), dtype=np.float64)
        gradient = -np.ones_like(activation)
        result = paper_edgecam_numpy(activation, gradient, normalize=False)
        np.testing.assert_array_equal(result, np.zeros((1, 2, 2)))

    def test_zero_denominator_is_finite(self):
        activation = np.zeros((1, 1, 2, 2), dtype=np.float64)
        gradient = np.zeros_like(activation)
        result = paper_edgecam_numpy(activation, gradient)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result, 0.0)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed locally")
class EdgeCAMTorchParityTests(unittest.TestCase):
    def test_numpy_and_torch_implementations_match(self):
        import torch

        from edge_finer_research.cams.edgecam import paper_edgecam_tensor

        rng = np.random.default_rng(8)
        activation = rng.normal(size=(2, 3, 4, 4)).astype(np.float32)
        gradient = rng.normal(size=(2, 3, 4, 4)).astype(np.float32)
        expected = paper_edgecam_numpy(activation, gradient)
        actual = paper_edgecam_tensor(torch.from_numpy(activation), torch.from_numpy(gradient)).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
