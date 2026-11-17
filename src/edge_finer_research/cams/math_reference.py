from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def normalize_maps(maps: NDArray[np.floating], epsilon: float = 1e-8) -> NDArray[np.float64]:
    maps = np.asarray(maps, dtype=np.float64)
    flat = maps.reshape(maps.shape[0], -1)
    minimum = flat.min(axis=1)[:, None, None]
    maximum = flat.max(axis=1)[:, None, None]
    span = maximum - minimum
    normalized = np.divide(maps - minimum, span, out=np.zeros_like(maps), where=span > epsilon)
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def paper_edgecam_numpy(
    activations: NDArray[np.floating],
    gradients: NDArray[np.floating],
    *,
    epsilon: float = 1e-8,
    apply_relu: bool = True,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Literal, testable implementation of EdgeCAM paper Equations 2--4.

    Inputs have shape ``[batch, channel, height, width]``. The denominator is
    ``2*g^2 + sum_ab(A_ab)*g^3`` at each channel/location, as printed in Eq. 4.
    This is intentionally kept distinct from any TorchCAM Grad-CAM++ variant.
    """
    activation = np.asarray(activations, dtype=np.float64)
    gradient = np.asarray(gradients, dtype=np.float64)
    if activation.shape != gradient.shape or activation.ndim != 4:
        raise ValueError("activations and gradients must have equal [B,C,H,W] shapes")
    gradient_squared = gradient.square() if hasattr(gradient, "square") else gradient**2
    denominator = 2.0 * gradient_squared + activation.sum(axis=(-2, -1), keepdims=True) * gradient**3
    alpha = np.divide(
        gradient_squared,
        denominator,
        out=np.zeros_like(gradient_squared),
        where=np.abs(denominator) > epsilon,
    )
    weights = alpha * np.maximum(gradient, 0.0)
    maps = np.sum(weights * activation, axis=1)
    if apply_relu:
        maps = np.maximum(maps, 0.0)
    maps = np.nan_to_num(maps, nan=0.0, posinf=0.0, neginf=0.0)
    return normalize_maps(maps, epsilon) if normalize else maps
