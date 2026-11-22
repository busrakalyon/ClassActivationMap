import json
import runpy
import shutil
import sys
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from edge_finer_research.data.manifest import ManifestRow, write_manifest


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "results" / "test_runtime"


@contextmanager
def writable_test_directory():
    path = TEST_TEMP_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=False)


class PixelOnlyEvaluationTests(unittest.TestCase):
    def test_fake_only_evaluation_omits_undefined_image_and_combined_metrics(self):
        with writable_test_directory() as root:
            manifest = root / "manifest.csv"
            mask_root = root / "masks"
            prediction_root = root / "predictions"
            output_root = root / "evaluation"
            mask_root.mkdir()
            prediction_root.mkdir()
            mask = np.array([[0, 255], [0, 255]], dtype=np.uint8)
            Image.fromarray(mask).save(mask_root / "sample.png")
            np.save(
                prediction_root / "sample__post_sop.npy",
                mask > 0,
                allow_pickle=False,
            )
            write_manifest(
                [
                    ManifestRow(
                        sample_id="sample",
                        dataset="fake_only",
                        split="test",
                        group_id="sample",
                        image_relpath="sample.tif",
                        mask_relpath="sample.png",
                        label=1,
                        width=2,
                        height=2,
                    )
                ],
                manifest,
            )
            script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
            arguments = [
                str(script),
                "--manifest",
                str(manifest),
                "--mask-root",
                str(mask_root),
                "--prediction-dir",
                str(prediction_root),
                "--output-dir",
                str(output_root),
                "--split",
                "test",
                "--prediction-suffix",
                "__post_sop.npy",
                "--pixel-scope",
                "fake",
                "--skip-image-metrics",
            ]
            with patch.object(sys, "argv", arguments):
                runpy.run_path(str(script), run_name="__main__")
            report = json.loads((output_root / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(report["pixel_f1"], 1.0)
            self.assertEqual(report["iou"], 1.0)
            self.assertIsNone(report["image_f1"])
            self.assertIsNone(report["image_auc"])
            self.assertIsNone(report["c_f1"])


if __name__ == "__main__":
    unittest.main()
