from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class StreamingMaxState:
    """Method-agnostic running state for gated epoch max-fusion."""

    map: NDArray[np.float32]
    valid_count: int
    processed_checkpoint_count: int
    logits: NDArray[np.float32]
    predicted_class: int
    fake_score: float


def initial_streaming_state(shape: tuple[int, int], *, num_classes: int = 2) -> StreamingMaxState:
    return StreamingMaxState(
        map=np.zeros(shape, dtype=np.float32),
        valid_count=0,
        processed_checkpoint_count=0,
        logits=np.zeros((num_classes,), dtype=np.float32),
        predicted_class=0,
        fake_score=0.0,
    )


def update_streaming_max(
    state: StreamingMaxState,
    candidate: NDArray[np.floating],
    *,
    valid: bool,
    logits: NDArray[np.floating],
    predicted_class: int,
    fake_score: float,
) -> StreamingMaxState:
    """Apply one epoch exactly once while preserving float32 CAM values."""
    values = np.asarray(candidate, dtype=np.float32)
    if values.shape != state.map.shape:
        raise ValueError(f"candidate shape {values.shape} differs from state {state.map.shape}")
    if not np.isfinite(values).all():
        raise ValueError("candidate CAM contains NaN or Inf")
    scores = np.asarray(logits, dtype=np.float32)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("logits must be a finite one-dimensional array")
    if not np.isfinite(fake_score):
        raise ValueError("fake score must be finite")
    if valid and state.valid_count == 0:
        fused = values.copy()
    elif valid:
        fused = np.maximum(state.map, values)
    else:
        fused = state.map.copy()
    return StreamingMaxState(
        map=np.asarray(fused, dtype=np.float32),
        valid_count=state.valid_count + int(valid),
        processed_checkpoint_count=state.processed_checkpoint_count + 1,
        logits=scores.copy(),
        predicted_class=int(predicted_class),
        fake_score=float(fake_score),
    )


def save_state_atomic(path: str | Path, state: StreamingMaxState, *, sample_id: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_id=np.asarray(sample_id),
            cam=state.map.astype(np.float32, copy=False),
            valid_count=np.asarray(state.valid_count, dtype=np.int64),
            processed_checkpoint_count=np.asarray(
                state.processed_checkpoint_count, dtype=np.int64
            ),
            logits=state.logits.astype(np.float32, copy=False),
            predicted_class=np.asarray(state.predicted_class, dtype=np.int64),
            fake_score=np.asarray(state.fake_score, dtype=np.float64),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def load_state(path: str | Path, *, expected_sample_id: str) -> StreamingMaxState:
    with np.load(Path(path), allow_pickle=False) as archive:
        sample_id = str(archive["sample_id"].item())
        if sample_id != expected_sample_id:
            raise RuntimeError(
                f"state sample ID mismatch: expected {expected_sample_id!r}, got {sample_id!r}"
            )
        cam = archive["cam"].astype(np.float32, copy=True)
        if cam.ndim != 2 or not np.isfinite(cam).all():
            raise RuntimeError(f"invalid running CAM in {path}")
        return StreamingMaxState(
            map=cam,
            valid_count=int(archive["valid_count"]),
            processed_checkpoint_count=int(archive["processed_checkpoint_count"]),
            logits=archive["logits"].astype(np.float32, copy=True),
            predicted_class=int(archive["predicted_class"]),
            fake_score=float(archive["fake_score"]),
        )


def write_json_atomic(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, destination)


def write_npy_atomic(path: str | Path, value: NDArray[Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
