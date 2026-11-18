import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed locally")
class DeterministicCAMPipelineTests(unittest.TestCase):
    def test_valid_nonzero_cams_are_included_in_max_fusion(self):
        from edge_finer_research.testing import run_deterministic_cam_control

        result = run_deterministic_cam_control("cpu")
        self.assertTrue(result["at_least_one_valid_fake_cam"])
        self.assertTrue(result["all_generated_cams_nonzero"])
        self.assertTrue(result["valid_maps_included_in_max_fusion"])
        self.assertEqual(result["valid_count"], 2)


if __name__ == "__main__":
    unittest.main()
