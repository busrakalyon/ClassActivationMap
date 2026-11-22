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

from edge_finer_research.data.columbia import (
    filename_from_legacy_entry,
    read_unique_vallist,
    threshold_mask_image,
    verify_existing_binary_png,
    write_binary_png_atomic,
)
from edge_finer_research.data.manifest import read_manifest


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "results" / "test_runtime"


@contextmanager
def writable_test_directory():
    path = TEST_TEMP_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=False)


class ColumbiaPreparationTests(unittest.TestCase):
    def test_end_to_end_builder_writes_exact_fake_only_manifest(self):
        with writable_test_directory() as root:
            image_dir = root / "4cam_splc"
            source_mask_dir = root / "mask"
            derived_mask_dir = root / "derived"
            image_dir.mkdir()
            source_mask_dir.mkdir()
            entries: list[str] = []
            image = np.zeros((3, 4, 3), dtype=np.uint8)
            source_mask = np.array(
                [[0, 0, 255, 255], [0, 0, 255, 255], [0, 0, 255, 255]],
                dtype=np.uint8,
            )
            for index in range(180):
                filename = f"sample_{index:03d}.tif"
                Image.fromarray(image).save(image_dir / filename)
                Image.fromarray(source_mask).save(
                    source_mask_dir / f"sample_{index:03d}.jpg", quality=100
                )
                entries.append(rf"C:\\legacy\\4cam_splc\\{filename}")
            vallist = root / "vallist.txt"
            vallist.write_text("\n".join(entries), encoding="utf-8")
            manifest = root / "columbia.csv"
            metadata = root / "metadata.json"
            arguments = [
                "prepare_columbia_hifi.py",
                "--dataset-root",
                str(root),
                "--vallist",
                str(vallist),
                "--output-mask-root",
                str(derived_mask_dir),
                "--output-manifest",
                str(manifest),
                "--metadata-output",
                str(metadata),
                "--expected-count",
                "180",
                "--source-revision",
                "test-revision",
            ]
            script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_columbia_hifi.py"
            with patch.object(sys, "argv", arguments):
                runpy.run_path(str(script), run_name="__main__")
            rows = read_manifest(manifest)
            self.assertEqual(len(rows), 180)
            self.assertEqual({row.split for row in rows}, {"test"})
            self.assertEqual({row.label for row in rows}, {1})
            self.assertTrue(all((derived_mask_dir / row.mask_relpath).is_file() for row in rows))
            audit_output = root / "audit.json"
            audit_arguments = [
                "audit_manifest.py",
                "--manifest",
                str(manifest),
                "--image-root",
                str(root),
                "--mask-root",
                str(derived_mask_dir),
                "--verify-hashes",
                "--require-binary-masks",
                "--require-mixed-masks",
                "--expected-rows",
                "180",
                "--expected-split",
                "test",
                "--expected-label",
                "1",
                "--output",
                str(audit_output),
            ]
            audit_script = Path(__file__).resolve().parents[1] / "scripts" / "audit_manifest.py"
            with patch.object(sys, "argv", audit_arguments):
                runpy.run_path(str(audit_script), run_name="__main__")
            audit = json.loads(audit_output.read_text(encoding="utf-8"))
            self.assertEqual(audit["error_count"], 0)
            self.assertEqual(audit["binary_mask_count"], 180)
            self.assertEqual(audit["mixed_mask_count"], 180)

    def test_legacy_absolute_paths_yield_only_tiff_filename(self):
        self.assertEqual(
            filename_from_legacy_entry(r"C:\old\Columbia\4cam_splc\sample.TIF"),
            "sample.TIF",
        )
        self.assertEqual(
            filename_from_legacy_entry("/old/Columbia/4cam_splc/sample.tiff"),
            "sample.tiff",
        )
        with self.assertRaises(ValueError):
            filename_from_legacy_entry(r"C:\old\Columbia\Thumbs.db")

    def test_vallist_requires_expected_unique_count(self):
        with writable_test_directory() as root:
            vallist = root / "vallist.txt"
            vallist.write_text("/old/a.tif\nC:\\old\\b.tif\n", encoding="utf-8")
            self.assertEqual(read_unique_vallist(vallist, expected_count=2), ["a.tif", "b.tif"])
            with self.assertRaises(ValueError):
                read_unique_vallist(vallist, expected_count=180)
            vallist.write_text("/old/a.tif\nC:\\other\\A.TIF\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_unique_vallist(vallist, expected_count=2)

    def test_threshold_is_grayscale_greater_equal_128_and_png_is_lossless(self):
        with writable_test_directory() as root:
            source = root / "source.png"
            output = root / "derived.png"
            Image.fromarray(np.array([[0, 127], [128, 255]], dtype=np.uint8)).save(source)
            binary, unique = threshold_mask_image(source)
            np.testing.assert_array_equal(
                binary, np.array([[0, 0], [255, 255]], dtype=np.uint8)
            )
            self.assertEqual(unique, {0, 255})
            write_binary_png_atomic(output, binary)
            verify_existing_binary_png(output, binary)


if __name__ == "__main__":
    unittest.main()
