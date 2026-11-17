from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    return float(numerator / denominator) if denominator else float(default)


def harmonic(left: float, right: float) -> float:
    return _safe_divide(2.0 * left * right, left + right)


@dataclass(frozen=True)
class ImageMetrics:
    sensitivity: float
    specificity: float
    paper_f1: float
    positive_class_f1: float
    auc: float
    tp: int
    tn: int
    fp: int
    fn: int


def binary_auc(labels: ArrayLike, scores: ArrayLike) -> float:
    """Mann-Whitney AUC with average ranks for tied scores."""
    truth = np.asarray(labels, dtype=np.int64).ravel()
    values = np.asarray(scores, dtype=np.float64).ravel()
    if truth.shape != values.shape:
        raise ValueError("labels and scores must have equal shapes")
    positive, negative = truth == 1, truth == 0
    n_positive, n_negative = int(positive.sum()), int(negative.sum())
    if not n_positive or not n_negative:
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end
    rank_sum = ranks[positive].sum()
    return float((rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative))


def image_metrics(labels: ArrayLike, predictions: ArrayLike, fake_scores: ArrayLike) -> ImageMetrics:
    truth = np.asarray(labels, dtype=np.int64).ravel()
    predicted = np.asarray(predictions, dtype=np.int64).ravel()
    if truth.shape != predicted.shape:
        raise ValueError("labels and predictions must have equal shapes")
    tp = int(np.sum((truth == 1) & (predicted == 1)))
    tn = int(np.sum((truth == 0) & (predicted == 0)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))
    sensitivity = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    positive_f1 = _safe_divide(2 * tp, 2 * tp + fp + fn)
    return ImageMetrics(
        sensitivity,
        specificity,
        harmonic(sensitivity, specificity),
        positive_f1,
        binary_auc(truth, fake_scores),
        tp,
        tn,
        fp,
        fn,
    )


@dataclass(frozen=True)
class PixelMetrics:
    f1: float
    iou: float
    tp: int
    tn: int
    fp: int
    fn: int


def pixel_metrics(
    ground_truth: ArrayLike, prediction: ArrayLike, *, empty_match_value: float = 1.0
) -> PixelMetrics:
    truth = np.asarray(ground_truth, dtype=bool)
    predicted = np.asarray(prediction, dtype=bool)
    if truth.shape != predicted.shape:
        raise ValueError("ground-truth and prediction masks must have equal shapes")
    tp = int(np.sum(truth & predicted))
    tn = int(np.sum(~truth & ~predicted))
    fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted))
    f1_denominator = 2 * tp + fp + fn
    union = tp + fp + fn
    f1 = _safe_divide(2 * tp, f1_denominator, empty_match_value)
    iou = _safe_divide(tp, union, empty_match_value)
    return PixelMetrics(f1, iou, tp, tn, fp, fn)


def combined_f1(image_f1: float, pixel_f1: float) -> float:
    """C-F1: harmonic mean of paper image-level F1 and pixel-level F1."""
    return harmonic(image_f1, pixel_f1)
