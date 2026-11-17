from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from PIL import Image


def _uint8_image(value: ArrayLike, *, binary: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if binary:
        return array.astype(bool).astype(np.uint8) * 255
    if array.dtype == np.uint8:
        return array
    array = np.nan_to_num(array.astype(np.float64))
    if array.ndim == 2:
        minimum, maximum = float(array.min()), float(array.max())
        array = (array - minimum) / (maximum - minimum) if maximum > minimum else np.zeros_like(array)
        return np.rint(array * 255).astype(np.uint8)
    if array.max(initial=0) <= 1.0:
        array = array * 255.0
    return np.clip(np.rint(array), 0, 255).astype(np.uint8)


def write_sample_artifacts(
    root: str | Path,
    sample_id: str,
    *,
    original: ArrayLike,
    ground_truth: ArrayLike,
    raw_cam: ArrayLike,
    epoch_fused_cam: ArrayLike,
    pre_sop_mask: ArrayLike,
    post_sop_mask: ArrayLike | None,
    metrics: dict[str, Any],
) -> Path:
    destination = Path(root) / sample_id
    destination.mkdir(parents=True, exist_ok=True)
    items = {
        "original.png": (_uint8_image(original), False),
        "ground_truth.png": (_uint8_image(ground_truth, binary=True), False),
        "raw_cam.png": (_uint8_image(raw_cam), False),
        "epoch_fused_cam.png": (_uint8_image(epoch_fused_cam), False),
        "pre_sop_mask.png": (_uint8_image(pre_sop_mask, binary=True), False),
    }
    if post_sop_mask is not None:
        items["post_sop_mask.png"] = (_uint8_image(post_sop_mask, binary=True), False)
    for filename, (array, _unused) in items.items():
        Image.fromarray(array).save(destination / filename)
    (destination / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return destination
