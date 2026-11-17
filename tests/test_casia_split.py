import unittest

from edge_finer_research.data.casia import assign_connected_group_ids, deterministic_validation_split
from edge_finer_research.data.manifest import ManifestRow, validate_manifest


def row(sample_id, sources):
    return ManifestRow(
        sample_id=sample_id,
        dataset="casia_v2",
        split="train",
        group_id="pending",
        image_relpath=f"{sample_id}.jpg",
        mask_relpath="mask.png" if len(sources) > 1 else "",
        label=1 if len(sources) > 1 else 0,
        source_ids=";".join(sources),
    )


class CasiaSplitTests(unittest.TestCase):
    def test_connected_sources_never_cross_split(self):
        rows = assign_connected_group_ids(
            [row("auth_a", ("aaa00001",)), row("auth_b", ("bbb00002",)), row("fake", ("aaa00001", "bbb00002"))]
        )
        self.assertEqual(len({item.group_id for item in rows}), 1)
        split = deterministic_validation_split(rows, validation_fraction=0.5, seed=42)
        self.assertEqual(len({item.split for item in split}), 1)
        self.assertFalse(validate_manifest(split))


if __name__ == "__main__":
    unittest.main()
