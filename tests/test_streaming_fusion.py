import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from edge_finer_research.cams.epoch_fusion import fuse_epoch_cams
from edge_finer_research.cams.streaming_fusion import (
    initial_streaming_state,
    load_state,
    save_state_atomic,
    update_streaming_max,
)


TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "results" / "test_runtime"


@contextmanager
def writable_test_directory():
    path = TEST_TEMP_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=False)


class StreamingMaxFusionTests(unittest.TestCase):
    def test_streaming_is_exactly_equal_to_materialized_float32_max(self):
        rng = np.random.default_rng(17)
        maps = [rng.random((5, 7), dtype=np.float32) for _ in range(6)]
        valid = [True, False, True, True, False, True]
        state = initial_streaming_state((5, 7))
        for index, (cam, keep) in enumerate(zip(maps, valid, strict=True)):
            state = update_streaming_max(
                state,
                cam,
                valid=keep,
                logits=np.array([-index, index], dtype=np.float32),
                predicted_class=index % 2,
                fake_score=index / 10,
            )
        materialized = fuse_epoch_cams(maps, valid=valid, method="max")
        np.testing.assert_array_equal(state.map, materialized.map.astype(np.float32))
        self.assertEqual(state.valid_count, sum(valid))
        self.assertEqual(state.processed_checkpoint_count, len(maps))

    def test_no_valid_epoch_produces_float32_zero_map(self):
        maps = [np.ones((3, 4), dtype=np.float32) for _ in range(3)]
        state = initial_streaming_state((3, 4))
        for cam in maps:
            state = update_streaming_max(
                state,
                cam,
                valid=False,
                logits=np.array([1.0, 0.0], dtype=np.float32),
                predicted_class=0,
                fake_score=0.25,
            )
        np.testing.assert_array_equal(state.map, np.zeros((3, 4), dtype=np.float32))
        self.assertEqual(state.map.dtype, np.float32)
        self.assertEqual(state.valid_count, 0)

    def test_first_valid_map_is_not_clipped_by_zero_initializer(self):
        state = update_streaming_max(
            initial_streaming_state((1, 2)),
            np.array([[-2.0, -1.0]], dtype=np.float32),
            valid=True,
            logits=np.array([0.0, 1.0], dtype=np.float32),
            predicted_class=1,
            fake_score=0.8,
        )
        np.testing.assert_array_equal(
            state.map, np.array([[-2.0, -1.0]], dtype=np.float32)
        )

    def test_atomic_lossless_state_roundtrip_preserves_resume_counter(self):
        state = update_streaming_max(
            initial_streaming_state((2, 2)),
            np.array([[0.1, 0.7], [0.3, 1.0]], dtype=np.float32),
            valid=True,
            logits=np.array([-2.0, 2.0], dtype=np.float32),
            predicted_class=1,
            fake_score=0.982,
        )
        with writable_test_directory() as directory:
            path = directory / "state.npz"
            save_state_atomic(path, state, sample_id="sample")
            restored = load_state(path, expected_sample_id="sample")
        np.testing.assert_array_equal(restored.map, state.map)
        np.testing.assert_array_equal(restored.logits, state.logits)
        self.assertEqual(restored.processed_checkpoint_count, 1)
        self.assertEqual(restored.valid_count, 1)
        self.assertEqual(restored.predicted_class, 1)

    def test_nonfinite_candidate_is_rejected(self):
        with self.assertRaises(ValueError):
            update_streaming_max(
                initial_streaming_state((1, 1)),
                np.array([[np.nan]], dtype=np.float32),
                valid=True,
                logits=np.array([0.0, 1.0], dtype=np.float32),
                predicted_class=1,
                fake_score=0.7,
            )


if __name__ == "__main__":
    unittest.main()
