from __future__ import annotations

from typing import Iterable

import numpy as np

from edge_finer_research.evaluation.metrics import pixel_metrics


def _binary_dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
        result = windows.any(axis=(-2, -1))
    return result


def _binary_erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
        result = windows.all(axis=(-2, -1))
    return result


def boundary_band(mask: np.ndarray, width: int) -> np.ndarray:
    """Symmetric morphological boundary using a 3x3 kernel for ``width`` iterations."""
    if width < 1:
        raise ValueError("boundary width must be at least one pixel")
    binary = np.asarray(mask, dtype=bool)
    return _binary_dilate(binary, width) & ~_binary_erode(binary, width)


def _region_mean(values: np.ndarray, region: np.ndarray) -> float:
    return float(values[region].mean()) if region.any() else float("nan")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    return float(numerator / denominator) if denominator != 0 else float("inf") if numerator > 0 else 0.0


def localization_statistics(cam: np.ndarray, ground_truth: np.ndarray, *, boundary_width: int) -> dict[str, float]:
    values = np.nan_to_num(np.asarray(cam, dtype=np.float64))
    truth = np.asarray(ground_truth, dtype=bool)
    if values.shape != truth.shape:
        raise ValueError("CAM and ground truth must have equal shapes")
    boundary = boundary_band(truth, boundary_width)
    inside_mean = _region_mean(values, truth)
    outside_mean = _region_mean(values, ~truth)
    boundary_mean = _region_mean(values, boundary)
    nonboundary_mean = _region_mean(values, ~boundary)
    total_energy = float(np.abs(values).sum())
    return {
        "inside_mean": inside_mean,
        "outside_mean": outside_mean,
        "inside_outside_ratio": _safe_ratio(inside_mean, outside_mean),
        "boundary_mean": boundary_mean,
        "nonboundary_mean": nonboundary_mean,
        "boundary_nonboundary_ratio": _safe_ratio(boundary_mean, nonboundary_mean),
        "energy_in_ground_truth_ratio": float(np.abs(values[truth]).sum() / total_energy) if total_energy else 0.0,
        "ground_truth_area_ratio": float(truth.mean()),
        "boundary_area_ratio": float(boundary.mean()),
    }


def safe_threshold_mask(cam: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold a CAM, keeping an invalid/all-zero CAM empty at every positive threshold."""
    if threshold <= 0:
        raise ValueError("diagnostic thresholds must be positive")
    values = np.nan_to_num(np.asarray(cam, dtype=np.float64))
    if values.size == 0 or float(values.max(initial=0.0)) <= 0.0:
        return np.zeros(values.shape, dtype=bool)
    return values >= threshold


def threshold_sweep(
    cams: Iterable[np.ndarray],
    masks: Iterable[np.ndarray],
    thresholds: Iterable[float],
) -> list[dict[str, float | str]]:
    heatmaps = list(cams)
    truth_masks = [np.asarray(mask, dtype=bool) for mask in masks]
    if not heatmaps or len(heatmaps) != len(truth_masks):
        raise ValueError("non-empty CAM and mask collections must have equal lengths")
    rows: list[dict[str, float | str]] = []
    for threshold in thresholds:
        predictions = [safe_threshold_mask(cam, threshold) for cam in heatmaps]
        measured = [pixel_metrics(truth, prediction) for truth, prediction in zip(truth_masks, predictions, strict=True)]
        rows.append(
            {
                "condition": "threshold",
                "threshold": float(threshold),
                "pixel_f1": float(np.mean([item.f1 for item in measured])),
                "iou": float(np.mean([item.iou for item in measured])),
                "predicted_area_ratio": float(np.mean([item.mean() for item in predictions])),
            }
        )
    for name, fill in (("all_image", True), ("empty", False)):
        predictions = [np.full(mask.shape, fill, dtype=bool) for mask in truth_masks]
        measured = [pixel_metrics(truth, prediction) for truth, prediction in zip(truth_masks, predictions, strict=True)]
        rows.append(
            {
                "condition": name,
                "threshold": float("nan"),
                "pixel_f1": float(np.mean([item.f1 for item in measured])),
                "iou": float(np.mean([item.iou for item in measured])),
                "predicted_area_ratio": float(fill),
            }
        )
    return rows
