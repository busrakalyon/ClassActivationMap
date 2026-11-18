from __future__ import annotations

import numpy as np

from .localization_analysis import boundary_band, localization_statistics


def _convolve3(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    padded = np.pad(image, 1, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    return np.einsum("ijxy,xy->ij", windows, kernel, optimize=True)


def sobel_edge_map(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float64)
    if values.ndim == 3:
        values = values[..., :3] @ np.array([0.299, 0.587, 0.114])
    if values.ndim != 2:
        raise ValueError("image must be HxW or HxWxC")
    gx = _convolve3(values, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64))
    gy = _convolve3(values, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64))
    magnitude = np.hypot(gx, gy)
    maximum = float(magnitude.max(initial=0.0))
    return magnitude / maximum if maximum > 0 else np.zeros_like(magnitude)


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError("arrays must have equal shapes")
    x, y = x - x.mean(), y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator else 0.0


def top_fraction_iou(left: np.ndarray, right: np.ndarray, *, fraction: float = 0.10) -> float:
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between zero and one")
    x, y = np.asarray(left), np.asarray(right)
    if x.shape != y.shape:
        raise ValueError("arrays must have equal shapes")
    left_active = x >= np.quantile(x, 1.0 - fraction)
    right_active = y >= np.quantile(y, 1.0 - fraction)
    union = np.count_nonzero(left_active | right_active)
    return float(np.count_nonzero(left_active & right_active) / union) if union else 1.0


def edge_comparison(cam: np.ndarray, sobel: np.ndarray, ground_truth: np.ndarray, *, boundary_width: int) -> dict[str, float]:
    if np.asarray(cam).shape != np.asarray(sobel).shape:
        raise ValueError("CAM and Sobel maps must have equal shapes")
    cam_local = localization_statistics(cam, ground_truth, boundary_width=boundary_width)
    sobel_local = localization_statistics(sobel, ground_truth, boundary_width=boundary_width)
    boundary = boundary_band(ground_truth, boundary_width)
    cam_boundary = float(np.asarray(cam)[boundary].mean()) if boundary.any() else float("nan")
    sobel_boundary = float(np.asarray(sobel)[boundary].mean()) if boundary.any() else float("nan")
    return {
        "cam_sobel_pearson": pearson_correlation(cam, sobel),
        "cam_sobel_top10_iou": top_fraction_iou(cam, sobel),
        "cam_boundary_mean": cam_boundary,
        "sobel_boundary_mean": sobel_boundary,
        "cam_boundary_nonboundary_ratio": cam_local["boundary_nonboundary_ratio"],
        "sobel_boundary_nonboundary_ratio": sobel_local["boundary_nonboundary_ratio"],
        "boundary_ratio_advantage_over_sobel": cam_local["boundary_nonboundary_ratio"] - sobel_local["boundary_nonboundary_ratio"],
    }
