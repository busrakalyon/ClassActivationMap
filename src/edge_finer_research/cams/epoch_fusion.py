from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EpochFusionResult:
    map: NDArray[np.float64]
    valid_count: int
    total_count: int


def fuse_epoch_cams(
    maps: Iterable[NDArray[np.floating]],
    *,
    valid: Iterable[bool] | None = None,
    method: Literal["max", "mean"] = "max",
) -> EpochFusionResult:
    items = [np.nan_to_num(np.asarray(item, dtype=np.float64)) for item in maps]
    if not items:
        raise ValueError("At least one epoch CAM is required")
    if any(item.shape != items[0].shape for item in items):
        raise ValueError("All epoch CAMs must have identical shapes")
    flags = list(valid) if valid is not None else [True] * len(items)
    if len(flags) != len(items):
        raise ValueError("valid flags and maps must have the same length")
    selected = [item for item, keep in zip(items, flags, strict=True) if keep]
    if not selected:
        return EpochFusionResult(np.zeros_like(items[0]), 0, len(items))
    stack = np.stack(selected)
    if method == "max":
        fused = stack.max(axis=0)
    elif method == "mean":
        fused = stack.mean(axis=0)
    else:
        raise ValueError(f"Unknown fusion method: {method}")
    return EpochFusionResult(fused, len(selected), len(items))
