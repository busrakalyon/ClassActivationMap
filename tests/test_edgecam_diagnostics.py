import importlib.util
import unittest

import numpy as np

from edge_finer_research.diagnostics.edge_analysis import pearson_correlation, sobel_edge_map
from edge_finer_research.diagnostics.edgecam_variants import compute_alpha_numpy
from edge_finer_research.diagnostics.localization_analysis import (
    boundary_band,
    localization_statistics,
    safe_threshold_mask,
    threshold_sweep,
)


class EdgeCAMDiagnosticMathTests(unittest.TestCase):
    def test_original_epsilon_suppresses_small_nonzero_gradient(self):
        activation = np.ones((1, 1, 2, 2), dtype=np.float64)
        gradient = np.full_like(activation, 1e-5)
        coarse = compute_alpha_numpy(activation, gradient, formula="original", epsilon=1e-8)
        fine = compute_alpha_numpy(activation, gradient, formula="original", epsilon=1e-12)
        self.assertTrue(coarse.suppressed_by_epsilon.all())
        np.testing.assert_array_equal(coarse.alpha, 0.0)
        self.assertFalse(fine.suppressed_by_epsilon.any())
        self.assertGreater(float(fine.alpha.min()), 0.49)

    def test_original_and_simplified_are_equivalent_away_from_special_cases(self):
        rng = np.random.default_rng(12)
        activation = rng.uniform(0.1, 1.0, size=(2, 3, 4, 4)).astype(np.float64)
        gradient = rng.uniform(0.01, 0.2, size=activation.shape).astype(np.float64)
        original = compute_alpha_numpy(activation, gradient, formula="original", epsilon=1e-16)
        simplified = compute_alpha_numpy(activation, gradient, formula="simplified", epsilon=1e-16)
        np.testing.assert_allclose(original.alpha, simplified.alpha, rtol=1e-12, atol=1e-12)

    def test_zero_gradient_is_safe_in_both_variants(self):
        activation = np.ones((1, 1, 1, 1), dtype=np.float64)
        gradient = np.zeros_like(activation)
        for formula in ("original", "simplified"):
            result = compute_alpha_numpy(activation, gradient, formula=formula, epsilon=1e-12)
            np.testing.assert_array_equal(result.alpha, 0.0)
            self.assertTrue(np.isfinite(result.alpha).all())

    def test_simplified_singular_denominator_is_safely_zero(self):
        activation = np.array([[[[1.0]]]])
        gradient = np.array([[[[-2.0]]]])
        result = compute_alpha_numpy(activation, gradient, formula="simplified", epsilon=1e-12)
        np.testing.assert_array_equal(result.denominator, 0.0)
        np.testing.assert_array_equal(result.alpha, 0.0)
        self.assertTrue(result.suppressed_by_epsilon.all())

    def test_near_singular_and_negative_denominators_are_distinguished(self):
        activation = np.array([[[[-2.0 + 1e-13]]]], dtype=np.float64)
        gradient = np.ones_like(activation)
        near = compute_alpha_numpy(activation, gradient, formula="simplified", epsilon=1e-12)
        self.assertTrue(near.suppressed_by_epsilon.all())
        negative = compute_alpha_numpy(
            np.array([[[[-3.0]]]]), gradient, formula="simplified", epsilon=1e-12
        )
        self.assertLess(float(negative.denominator.item()), 0.0)
        self.assertLess(float(negative.alpha.item()), 0.0)

    def test_float32_and_float64_make_same_epsilon_decision_for_representable_case(self):
        for dtype in (np.float32, np.float64):
            activation = np.ones((1, 1, 2, 2), dtype=dtype)
            gradient = np.full_like(activation, 1e-5)
            result = compute_alpha_numpy(activation, gradient, formula="original", epsilon=1e-8)
            self.assertTrue(result.suppressed_by_epsilon.all())

    def test_nonfinite_inputs_do_not_escape_as_alpha(self):
        activation = np.ones((1, 1, 1, 3), dtype=np.float64)
        gradient = np.array([[[[np.nan, np.inf, -np.inf]]]])
        result = compute_alpha_numpy(activation, gradient, formula="original", epsilon=1e-8)
        self.assertTrue(np.isfinite(result.alpha).all())


class LocalizationDiagnosticTests(unittest.TestCase):
    def test_boundary_band_and_localization_metrics(self):
        mask = np.zeros((7, 7), dtype=bool)
        mask[2:5, 2:5] = True
        boundary = boundary_band(mask, 1)
        self.assertTrue(boundary.any())
        self.assertFalse(boundary[3, 3])
        cam = boundary.astype(np.float64)
        measured = localization_statistics(cam, mask, boundary_width=1)
        self.assertGreater(measured["boundary_nonboundary_ratio"], 1.0)

    def test_zero_cam_stays_empty_at_low_positive_threshold(self):
        cam = np.zeros((3, 3), dtype=np.float64)
        self.assertFalse(safe_threshold_mask(cam, 0.001).any())
        rows = threshold_sweep([cam], [np.eye(3, dtype=bool)], [0.001])
        self.assertEqual(rows[0]["predicted_area_ratio"], 0.0)
        self.assertEqual({row["condition"] for row in rows}, {"threshold", "all_image", "empty"})

    def test_sobel_and_correlation_are_finite(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[:, 4:] = 255
        edge = sobel_edge_map(image)
        self.assertGreater(float(edge.max()), 0.0)
        self.assertAlmostEqual(pearson_correlation(edge, edge), 1.0)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed locally")
class TorchDiagnosticTests(unittest.TestCase):
    def test_batch_normalization_is_per_sample_and_negative_gradient_contributes_nothing(self):
        import torch

        from edge_finer_research.cams.edgecam import paper_edgecam_tensor
        from edge_finer_research.diagnostics.edgecam_variants import compute_edgecam_diagnostic_tensor

        activation = torch.ones((2, 1, 2, 2), dtype=torch.float64)
        activation[1, 0, 1, 1] = 2.0
        gradient = torch.ones_like(activation)
        result = compute_edgecam_diagnostic_tensor(
            activation, gradient, formula="simplified", epsilon=1e-12
        )
        self.assertTrue(bool(result.constant_map[0]))
        self.assertFalse(bool(result.constant_map[1]))
        np.testing.assert_array_equal(result.cam_normalized[0].numpy(), 0.0)

        negative = compute_edgecam_diagnostic_tensor(
            activation, -gradient, formula="simplified", epsilon=1e-12, normalize=False
        )
        np.testing.assert_array_equal(negative.cam_post_relu.numpy(), 0.0)

        baseline = paper_edgecam_tensor(activation, gradient, epsilon=1e-8)
        diagnostic = compute_edgecam_diagnostic_tensor(
            activation,
            gradient,
            formula="original",
            epsilon=1e-8,
            normalization_epsilon=1e-8,
        )
        torch.testing.assert_close(diagnostic.cam_normalized, baseline)

    def test_individual_and_batched_gradients_do_not_mix(self):
        import torch

        from edge_finer_research.diagnostics.edgecam_variants import extract_activation_gradients

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 2, 1, bias=False)
                self.head = torch.nn.Linear(2, 2, bias=False)

            def forward_logits(self, images):
                activation = self.conv(images).relu()
                return self.head(activation.mean(dim=(-2, -1)))

        torch.manual_seed(3)
        model = TinyModel().eval()
        images = torch.randn(4, 1, 3, 3)
        batch = extract_activation_gradients(model, model.conv, images, 1)
        individual = [extract_activation_gradients(model, model.conv, image[None], 1) for image in images]
        for batch_values, offset in zip(batch, range(3), strict=True):
            expected = torch.cat([item[offset] for item in individual])
            torch.testing.assert_close(batch_values, expected, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
