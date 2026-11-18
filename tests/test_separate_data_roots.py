import importlib.util
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

from edge_finer_research.data.manifest import ManifestRow, validate_manifest


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "results" / "test_runtime"


@contextmanager
def writable_test_directory():
    path = TEST_TEMP_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=False)


def fake_row() -> ManifestRow:
    return ManifestRow(
        sample_id="fake-1",
        dataset="casia_v2",
        split="train",
        group_id="source-1",
        image_relpath="Tp/fake.jpg",
        mask_relpath="fake_gt.png",
        label=1,
    )


class SeparateRootValidationTests(unittest.TestCase):
    def test_image_and_mask_files_validate_under_different_roots(self):
        with writable_test_directory() as root:
            image_root, mask_root = root / "images", root / "masks"
            (image_root / "Tp").mkdir(parents=True)
            mask_root.mkdir()
            (image_root / "Tp/fake.jpg").write_bytes(b"image-placeholder")
            (mask_root / "fake_gt.png").write_bytes(b"mask-placeholder")

            issues = validate_manifest(
                [fake_row()], image_root=image_root, mask_root=mask_root
            )

            self.assertEqual(issues, [])

    def test_mask_root_must_be_explicit_for_file_validation(self):
        with writable_test_directory() as image_root:
            (image_root / "Tp").mkdir()
            (image_root / "Tp/fake.jpg").write_bytes(b"image-placeholder")

            issues = validate_manifest([fake_row()], image_root=image_root)

            self.assertTrue(any("mask_root is required" in issue.message for issue in issues))
            self.assertFalse(any("missing mask:" in issue.message for issue in issues))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed locally")
class WeakSupervisionDatasetTests(unittest.TestCase):
    def test_classifier_dataset_does_not_open_manifest_mask(self):
        import torch
        from PIL import Image

        from edge_finer_research.data.dataset import ManifestClassificationDataset

        with writable_test_directory() as image_root:
            (image_root / "Tp").mkdir()
            Image.new("RGB", (4, 4), color="white").save(image_root / "Tp/fake.jpg")
            dataset = ManifestClassificationDataset(
                [fake_row()],
                image_root,
                transform=lambda _image: torch.zeros((3, 4, 4)),
            )

            image, label, sample_id = dataset[0]

            self.assertEqual(tuple(image.shape), (3, 4, 4))
            self.assertEqual(label.item(), 1)
            self.assertEqual(sample_id, "fake-1")


if __name__ == "__main__":
    unittest.main()
