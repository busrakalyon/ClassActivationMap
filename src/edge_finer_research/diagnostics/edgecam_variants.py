from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


Formula = Literal["original", "simplified"]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    formula: Formula
    epsilon: float


@dataclass(frozen=True)
class AlphaNumpyResult:
    alpha: np.ndarray
    raw_alpha: np.ndarray
    denominator: np.ndarray
    suppressed_by_epsilon: np.ndarray


@dataclass(frozen=True)
class DiagnosticCAMResult:
    alpha: Any
    raw_alpha: Any
    denominator: Any
    suppressed_by_epsilon: Any
    weights: Any
    channel_contributions: Any
    cam_pre_relu: Any
    cam_post_relu: Any
    cam_normalized: Any
    constant_map: Any


def default_variant_specs() -> list[VariantSpec]:
    return [
        *(VariantSpec(f"original_eps_{epsilon:.0e}", "original", epsilon) for epsilon in (1e-8, 1e-10, 1e-12, 1e-16)),
        VariantSpec("simplified_eps_1e-12", "simplified", 1e-12),
    ]


def compute_alpha_numpy(
    activations: np.ndarray,
    gradients: np.ndarray,
    *,
    formula: Formula,
    epsilon: float,
) -> AlphaNumpyResult:
    """Reference alpha calculation with diagnostics, separate from baseline code."""
    activation = np.asarray(activations)
    gradient = np.asarray(gradients)
    if activation.shape != gradient.shape or activation.ndim != 4:
        raise ValueError("activations and gradients must have equal [B,C,H,W] shapes")
    spatial_sum = activation.sum(axis=(-2, -1), keepdims=True)
    nonzero_gradient = gradient != 0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        if formula == "original":
            gradient_squared = gradient**2
            denominator = 2.0 * gradient_squared + spatial_sum * gradient**3
            raw_alpha = gradient_squared / denominator
            usable = np.isfinite(denominator) & (np.abs(denominator) > epsilon)
            suppressed = nonzero_gradient & np.isfinite(denominator) & ~usable
        elif formula == "simplified":
            denominator = 2.0 + spatial_sum * gradient
            raw_alpha = 1.0 / denominator
            usable = nonzero_gradient & np.isfinite(denominator) & (np.abs(denominator) > epsilon)
            suppressed = nonzero_gradient & np.isfinite(denominator) & ~usable
        else:
            raise ValueError(f"Unknown formula: {formula}")
    alpha = np.where(usable & np.isfinite(raw_alpha), raw_alpha, 0.0)
    return AlphaNumpyResult(alpha, raw_alpha, denominator, suppressed)


def normalize_per_sample_tensor(maps: Any, *, epsilon: float) -> tuple[Any, Any]:
    """Per-image normalization matching ``paper_edgecam_tensor``."""
    flat = maps.flatten(1)
    minimum = flat.amin(dim=1).view(-1, 1, 1)
    span = (flat.amax(dim=1) - flat.amin(dim=1)).view(-1, 1, 1)
    constant = span[:, 0, 0] <= epsilon
    normalized = (maps - minimum) / span.clamp_min(epsilon)
    normalized = normalized.masked_fill(constant[:, None, None], 0.0)
    return normalized, constant


def compute_edgecam_diagnostic_tensor(
    activations: Any,
    gradients: Any,
    *,
    formula: Formula,
    epsilon: float,
    normalization_epsilon: float = 1e-8,
    final_relu: bool = True,
    normalize: bool = True,
) -> DiagnosticCAMResult:
    """Compute a diagnostic variant without modifying the baseline implementation."""
    import torch

    if activations.shape != gradients.shape or activations.ndim != 4:
        raise ValueError("activations and gradients must have equal [B,C,H,W] shapes")
    spatial_sum = activations.sum(dim=(-2, -1), keepdim=True)
    nonzero_gradient = gradients != 0
    if formula == "original":
        gradient_squared = gradients.square()
        denominator = 2.0 * gradient_squared + spatial_sum * gradients.pow(3)
        raw_alpha = gradient_squared / denominator
    elif formula == "simplified":
        denominator = 2.0 + spatial_sum * gradients
        raw_alpha = denominator.reciprocal()
    else:
        raise ValueError(f"Unknown formula: {formula}")
    usable = nonzero_gradient & torch.isfinite(denominator) & (denominator.abs() > epsilon)
    suppressed = nonzero_gradient & torch.isfinite(denominator) & ~usable
    alpha = torch.where(usable & torch.isfinite(raw_alpha), raw_alpha, torch.zeros_like(raw_alpha))
    weights = alpha * gradients.clamp_min(0.0)
    contributions = weights * activations
    pre_relu = contributions.sum(dim=1)
    post_relu = pre_relu.relu() if final_relu else pre_relu
    post_relu = torch.nan_to_num(post_relu)
    if normalize:
        normalized, constant = normalize_per_sample_tensor(post_relu, epsilon=normalization_epsilon)
    else:
        normalized = post_relu
        constant = torch.zeros(post_relu.shape[0], dtype=torch.bool, device=post_relu.device)
    return DiagnosticCAMResult(
        alpha,
        raw_alpha,
        denominator,
        suppressed,
        weights,
        contributions,
        pre_relu,
        post_relu,
        normalized,
        constant,
    )


def extract_activation_gradients(model: Any, target_layer: Any, images: Any, target_class: int) -> tuple[Any, Any, Any]:
    """One float32, label-free forward/gradient pass; no AMP and no parameter-grad accumulation."""
    import torch

    captured: list[Any] = []
    hook = target_layer.register_forward_hook(lambda _module, _inputs, output: captured.append(output))
    try:
        model.eval()
        with torch.enable_grad():
            logits = model.forward_logits(images)
            if len(captured) != 1:
                raise RuntimeError(f"Target layer executed {len(captured)} times; expected exactly once")
            activation = captured[0]
            selected = logits[:, target_class].sum()
            gradients = torch.autograd.grad(selected, activation, retain_graph=False, create_graph=False)[0]
    finally:
        hook.remove()
    return logits.detach(), activation.detach(), gradients.detach()
