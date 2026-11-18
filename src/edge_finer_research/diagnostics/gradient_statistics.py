from __future__ import annotations

from typing import Any

import numpy as np


PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


def distribution_statistics(values: Any, *, prefix: str = "") -> dict[str, float | int]:
    array = np.asarray(values).ravel()
    finite = array[np.isfinite(array)]
    result: dict[str, float | int] = {
        f"{prefix}count": int(array.size),
        f"{prefix}nan_count": int(np.isnan(array).sum()),
        f"{prefix}posinf_count": int(np.isposinf(array).sum()),
        f"{prefix}neginf_count": int(np.isneginf(array).sum()),
    }
    if not finite.size:
        for key in ("min", "max", "mean", "std", "abs_mean"):
            result[f"{prefix}{key}"] = float("nan")
        for percentile in PERCENTILES:
            result[f"{prefix}p{percentile}"] = float("nan")
        return result
    result.update(
        {
            f"{prefix}min": float(finite.min()),
            f"{prefix}max": float(finite.max()),
            f"{prefix}mean": float(finite.mean()),
            f"{prefix}std": float(finite.std()),
            f"{prefix}abs_mean": float(np.abs(finite).mean()),
        }
    )
    result.update({f"{prefix}p{p}": float(np.percentile(finite, p)) for p in PERCENTILES})
    return result


def gradient_statistics(gradient: Any) -> dict[str, float | int]:
    array = np.asarray(gradient)
    count = max(array.size, 1)
    return {
        **distribution_statistics(array),
        "zero_ratio": float(np.count_nonzero(array == 0) / count),
        "positive_ratio": float(np.count_nonzero(array > 0) / count),
        "negative_ratio": float(np.count_nonzero(array < 0) / count),
    }


def alpha_statistics(
    alpha: Any,
    raw_alpha: Any,
    suppressed: Any,
    gradient: Any,
    denominator: Any,
) -> dict[str, float | int]:
    values = np.asarray(alpha)
    raw = np.asarray(raw_alpha)
    suppressed_mask = np.asarray(suppressed, dtype=bool)
    gradients = np.asarray(gradient)
    denominators = np.asarray(denominator)
    positive = gradients > 0
    count = max(values.size, 1)
    positive_count = max(int(positive.sum()), 1)
    return {
        **distribution_statistics(values),
        **distribution_statistics(denominators, prefix="denominator_"),
        "epsilon_suppressed_ratio": float(suppressed_mask.sum() / count),
        "positive_gradient_epsilon_suppressed_ratio": float((suppressed_mask & positive).sum() / positive_count),
        "negative_alpha_ratio": float(np.count_nonzero(values < 0) / count),
        "negative_denominator_ratio": float(np.count_nonzero(denominators < 0) / count),
        "zero_alpha_ratio": float(np.count_nonzero(values == 0) / count),
        "raw_alpha_nan_count": int(np.isnan(raw).sum()),
        "raw_alpha_posinf_count": int(np.isposinf(raw).sum()),
        "raw_alpha_neginf_count": int(np.isneginf(raw).sum()),
    }


def cam_statistics(result: Any, *, active_thresholds: tuple[float, ...]) -> dict[str, float | int]:
    pre = np.asarray(result.cam_pre_relu)
    post = np.asarray(result.cam_post_relu)
    normalized = np.asarray(result.cam_normalized)
    contributions = np.asarray(result.channel_contributions)
    negative_magnitude = np.maximum(-contributions, 0).sum()
    absolute_magnitude = np.abs(contributions).sum()
    output: dict[str, float | int] = {
        **distribution_statistics(pre, prefix="pre_relu_"),
        **distribution_statistics(post, prefix="post_relu_pre_normalization_"),
        **distribution_statistics(normalized, prefix="normalized_"),
        "pre_relu_negative_pixel_ratio": float(np.mean(pre < 0)),
        "post_relu_zero_pixel_ratio": float(np.mean(post == 0)),
        "normalized_zero_pixel_ratio": float(np.mean(normalized == 0)),
        "negative_channel_contribution_fraction": float(negative_magnitude / absolute_magnitude) if absolute_magnitude else 0.0,
    }
    for threshold in active_thresholds:
        output[f"normalized_active_ratio_at_{threshold:g}"] = float(np.mean(normalized >= threshold))
    return output
