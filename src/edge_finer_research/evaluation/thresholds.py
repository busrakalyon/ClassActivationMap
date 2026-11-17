from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .metrics import pixel_metrics


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    pixel_f1: float
    iou: float
    scores: tuple[tuple[float, float, float], ...]


def select_pixel_threshold(
    heatmaps: Iterable[np.ndarray],
    masks: Iterable[np.ndarray],
    candidates: Iterable[float],
) -> ThresholdSelection:
    maps, truths = list(heatmaps), list(masks)
    values = sorted(set(float(value) for value in candidates))
    if not maps or len(maps) != len(truths) or not values:
        raise ValueError("Equal non-empty heatmaps/masks and candidate thresholds are required")
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("Threshold candidates must be in [0,1]")
    scores: list[tuple[float, float, float]] = []
    for threshold in values:
        metrics = [pixel_metrics(mask, heatmap >= threshold) for heatmap, mask in zip(maps, truths, strict=True)]
        scores.append(
            (threshold, float(np.mean([item.f1 for item in metrics])), float(np.mean([item.iou for item in metrics])))
        )
    # Maximize validation P-F1, then IoU; prefer the smaller threshold for an exact tie.
    best = max(scores, key=lambda item: (item[1], item[2], -item[0]))
    return ThresholdSelection(best[0], best[1], best[2], tuple(scores))
