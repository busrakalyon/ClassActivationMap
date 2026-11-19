import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from edge_finer_research.proposals.mcg import (
    MCGProposalCache,
    MCGNpzProposalProvider,
    load_mcg_cache,
    parse_matlab_intermediate,
    validate_reusable_cache,
    write_mcg_cache_atomic,
)
from edge_finer_research.proposals.sop import proposal_edge, select_object_proposal


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "results" / "test_runtime"


@contextmanager
def writable_test_directory():
    path = TEST_TEMP_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=False)


def example_cache(height: int = 7, width: int = 9, count: int = 50) -> MCGProposalCache:
    masks = np.zeros((height, width, count), dtype=np.uint8)
    for index in range(count):
        masks[index % height, index % width, index] = 1
    return MCGProposalCache(
        masks,
        np.linspace(1.0, 0.0, count, dtype=np.float64),
        np.arange(1, count + 1, dtype=np.int32),
    )


class MCGProposalCacheTests(unittest.TestCase):
    def test_missing_cache_is_pending_for_resume(self):
        with writable_test_directory() as root:
            self.assertFalse(
                validate_reusable_cache(
                    root / "missing.npz",
                    root / "missing.json",
                    sample_id="missing",
                    image_sha256="abc",
                    expected_height=7,
                    expected_width=9,
                )
            )

    def test_npz_contract_preserves_fifty_ranked_binary_masks(self):
        with writable_test_directory() as root:
            path = root / "sample.npz"
            write_mcg_cache_atomic(path, example_cache())
            loaded = load_mcg_cache(
                path, expected_height=7, expected_width=9, expected_count=50
            )
            self.assertEqual(loaded.masks_hwk.shape, (7, 9, 50))
            self.assertEqual(loaded.masks_hwk.dtype, np.uint8)
            np.testing.assert_array_equal(loaded.ranks, np.arange(1, 51))

    def test_npz_contract_rejects_wrong_proposal_count(self):
        with writable_test_directory() as root:
            path = root / "sample.npz"
            write_mcg_cache_atomic(path, example_cache(count=49))
            with self.assertRaises(ValueError):
                load_mcg_cache(
                    path, expected_height=7, expected_width=9, expected_count=50
                )

    def test_matlab_column_major_intermediate_is_parsed_exactly(self):
        with writable_test_directory() as root:
            cache = example_cache(height=3, width=4)
            cache.masks_hwk.ravel(order="F").tofile(root / "masks_uint8_f_order.bin")
            cache.scores.astype("<f8").tofile(root / "scores_float64_le.bin")
            metadata = {
                "generator": "official_mcg_reconstructed",
                "sample_id": "sample",
                "height": 3,
                "width": 4,
                "proposal_count_total": 5044,
                "saved_top_k": 50,
                "mcg_mode": "accurate",
                "matlab_version": "test",
            }
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            (root / "completed.json").write_text('{"status":"completed"}', encoding="utf-8")
            parsed, parsed_metadata = parse_matlab_intermediate(
                root, expected_height=3, expected_width=4, expected_count=50
            )
            np.testing.assert_array_equal(parsed.masks_hwk, cache.masks_hwk)
            np.testing.assert_array_equal(parsed.scores, cache.scores)
            self.assertEqual(parsed_metadata["proposal_count_total"], 5044)

    def test_valid_cache_is_reused_and_changed_image_is_rejected(self):
        with writable_test_directory() as root:
            cache_path, metadata_path = root / "sample.npz", root / "sample.json"
            write_mcg_cache_atomic(cache_path, example_cache())
            metadata = {
                "generator": "official_mcg_reconstructed",
                "sample_id": "sample",
                "image_sha256": "abc",
                "image_height": 7,
                "image_width": 9,
                "saved_top_k": 50,
                "mcg_mode": "accurate",
                "proposal_count_total": 5044,
            }
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            self.assertTrue(
                validate_reusable_cache(
                    cache_path,
                    metadata_path,
                    sample_id="sample",
                    image_sha256="abc",
                    expected_height=7,
                    expected_width=9,
                )
            )
            with self.assertRaises(ValueError):
                validate_reusable_cache(
                    cache_path,
                    metadata_path,
                    sample_id="sample",
                    image_sha256="changed",
                    expected_height=7,
                    expected_width=9,
                )

    def test_mcg_cache_provider_and_sop_scoring_smoke(self):
        with writable_test_directory() as root:
            cache = example_cache(height=9, width=9)
            cache.masks_hwk[:, :, :] = 0
            cache.masks_hwk[3:6, 3:6, 7] = 1
            write_mcg_cache_atomic(root / "sample.npz", cache)
            provider = MCGNpzProposalProvider(root)
            proposals = provider.get_for_shape("sample", height=9, width=9)
            heatmap = proposal_edge(proposals)[7].astype(np.float32)
            result = select_object_proposal(heatmap, proposals)
            self.assertEqual(result.proposal_index, 7)
            np.testing.assert_array_equal(result.mask, proposals[7])


if __name__ == "__main__":
    unittest.main()
