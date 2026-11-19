import unittest

import numpy as np
from PIL import Image

from edge_finer_research.data.manifest import ManifestRow
from edge_finer_research.data.overlap import (
    ImageFingerprint,
    decoded_rgb_sha256,
    exact_cross_matches,
    perceptual_hash,
    perceptual_cross_matches,
    source_cross_matches,
)


def row(sample_id: str, relpath: str, source_ids: str, *, split: str = "train") -> ManifestRow:
    return ManifestRow(
        sample_id=sample_id,
        dataset="test",
        split=split,
        group_id=source_ids,
        image_relpath=relpath,
        mask_relpath="",
        label=0,
        source_ids=source_ids,
        width=16,
        height=16,
    )


class DatasetOverlapTests(unittest.TestCase):
    def test_distinguishes_file_bytes_from_decoded_rgb(self):
        image = Image.fromarray(np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3))
        rgb_hash = decoded_rgb_sha256(image)
        phash = perceptual_hash(image)
        left = ImageFingerprint(row("left", "left.png", "cha00001"), "encoded-a", rgb_hash, phash)
        right = ImageFingerprint(
            row("right", "right.bmp", "cha00002", split="test"),
            "encoded-b",
            rgb_hash,
            phash,
        )
        self.assertEqual(exact_cross_matches([left], [right], attribute="file_sha256"), [])
        self.assertEqual(len(exact_cross_matches([left], [right], attribute="rgb_sha256")), 1)

    def test_perceptual_matches_are_candidates_with_distance(self):
        pixels = np.zeros((32, 32, 3), dtype=np.uint8)
        pixels[8:24, 8:24] = 255
        image = Image.fromarray(pixels)
        resized = image.resize((64, 64))
        left = ImageFingerprint(
            row("left", "left.png", "cha00001"),
            "encoded-a",
            decoded_rgb_sha256(image),
            perceptual_hash(image),
        )
        right = ImageFingerprint(
            row("right", "right.png", "cha00002", split="test"),
            "encoded-b",
            decoded_rgb_sha256(resized),
            perceptual_hash(resized),
        )
        matches = perceptual_cross_matches([left], [right], max_distance=12)
        self.assertEqual(len(matches), 1)
        self.assertLessEqual(matches[0][2], 12)

    def test_shared_sources_link_different_manipulations(self):
        left = ImageFingerprint(
            row("left", "left.png", "cha00001;nat00002"), "a", "b", 0
        )
        right = ImageFingerprint(
            row("right", "right.png", "cha00001;txt00003", split="test"), "c", "d", 1
        )
        matches = source_cross_matches([left], [right])
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][2], ("cha00001",))


if __name__ == "__main__":
    unittest.main()
