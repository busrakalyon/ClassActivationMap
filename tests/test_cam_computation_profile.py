import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed locally")
class ScientificCAMProfileTests(unittest.TestCase):
    def test_profile_is_applied_and_runtime_verified(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        from edge_finer_research.cams.computation_profile import (
            apply_scientific_cam_profile,
            verify_scientific_cam_runtime,
        )

        previous_matmul = torch.backends.cuda.matmul.allow_tf32
        previous_cudnn = torch.backends.cudnn.allow_tf32
        previous_precision = torch.get_float32_matmul_precision()
        try:
            profile = apply_scientific_cam_profile(torch.device("cuda"), batch_size=1)
            model = torch.nn.Linear(2, 2).cuda().float().eval()
            images = torch.ones((1, 2), device="cuda", dtype=torch.float32)
            runtime = verify_scientific_cam_runtime(model, images)
            self.assertTrue(profile["settings_verified"])
            self.assertFalse(profile["actual_backend"]["cuda_matmul_allow_tf32"])
            self.assertFalse(profile["actual_backend"]["cudnn_allow_tf32"])
            self.assertTrue(runtime["model_eval_verified"])
            self.assertTrue(runtime["amp_verified_disabled"])
        finally:
            torch.set_float32_matmul_precision(previous_precision)
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul
            torch.backends.cudnn.allow_tf32 = previous_cudnn


if __name__ == "__main__":
    unittest.main()
