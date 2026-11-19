from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .metrics import pixel_metrics


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    pixel_f1: float
    iou: float
    predicted_area_ratio: float
    ground_truth_area_ratio: float


def threshold_mask(cam: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(cam)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.floating):
        raise ValueError("CAM must be a two-dimensional floating-point array")
    if not np.isfinite(values).all():
        raise ValueError("CAM contains NaN or Inf")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    # A strictly positive threshold preserves an all-zero CAM as an empty mask.
    return values >= threshold


def measure_thresholds(
    cams: Iterable[np.ndarray],
    masks: Iterable[np.ndarray],
    thresholds: Iterable[float],
) -> tuple[ThresholdMetrics, ...]:
    maps, truths = list(cams), [np.asarray(mask, dtype=bool) for mask in masks]
    values = sorted(set(float(value) for value in thresholds))
    if not maps or len(maps) != len(truths) or not values:
        raise ValueError("Equal non-empty CAM/mask collections and thresholds are required")
    output: list[ThresholdMetrics] = []
    for threshold in values:
        predictions = [threshold_mask(cam, threshold) for cam in maps]
        if any(
            prediction.shape != truth.shape
            for prediction, truth in zip(predictions, truths, strict=True)
        ):
            raise ValueError("CAM and ground-truth shapes differ")
        metrics = [
            pixel_metrics(truth, prediction)
            for prediction, truth in zip(predictions, truths, strict=True)
        ]
        output.append(
            ThresholdMetrics(
                threshold=threshold,
                pixel_f1=float(np.mean([item.f1 for item in metrics])),
                iou=float(np.mean([item.iou for item in metrics])),
                predicted_area_ratio=float(np.mean([item.mean() for item in predictions])),
                ground_truth_area_ratio=float(np.mean([item.mean() for item in truths])),
            )
        )
    return tuple(output)


def reference_masks(masks: Iterable[np.ndarray]) -> dict[str, dict[str, float]]:
    truths = [np.asarray(mask, dtype=bool) for mask in masks]
    if not truths:
        raise ValueError("At least one ground-truth mask is required")
    output: dict[str, dict[str, float]] = {}
    for name, fill in (("empty", False), ("full_image", True)):
        predictions = [np.full(truth.shape, fill, dtype=bool) for truth in truths]
        metrics = [
            pixel_metrics(truth, prediction)
            for truth, prediction in zip(truths, predictions, strict=True)
        ]
        output[name] = {
            "pixel_f1": float(np.mean([item.f1 for item in metrics])),
            "iou": float(np.mean([item.iou for item in metrics])),
            "predicted_area_ratio": float(fill),
            "ground_truth_area_ratio": float(np.mean([item.mean() for item in truths])),
        }
    return output
