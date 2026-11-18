from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np


CUDA_PROFILES = {
    "tf32_on_deterministic_off": {"tf32": True, "deterministic": False},
    "tf32_off_deterministic_off": {"tf32": False, "deterministic": False},
    "tf32_on_deterministic_on": {"tf32": True, "deterministic": True},
    "tf32_off_deterministic_on": {"tf32": False, "deterministic": True},
}


def pearson(left: Any, right: Any) -> float:
    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError("arrays must have equal shapes")
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator == 0:
        return 1.0 if np.array_equal(np.asarray(left), np.asarray(right)) else 0.0
    return float(np.dot(x, y) / denominator)


def mask_iou(left: Any, right: Any, threshold: float) -> float:
    first = np.asarray(left) >= threshold
    second = np.asarray(right) >= threshold
    union = int(np.count_nonzero(first | second))
    return float(np.count_nonzero(first & second) / union) if union else 1.0


def numerical_comparison(left: Any, right: Any, *, atol: float, rtol: float) -> dict[str, float | bool]:
    first = np.asarray(left)
    second = np.asarray(right)
    if first.shape != second.shape:
        raise ValueError(f"array shapes differ: {first.shape} != {second.shape}")
    difference = np.abs(first.astype(np.float64) - second.astype(np.float64))
    return {
        "allclose": bool(np.allclose(first, second, atol=atol, rtol=rtol, equal_nan=False)),
        "max_abs_difference": float(difference.max(initial=0.0)),
        "mean_abs_difference": float(difference.mean()) if difference.size else 0.0,
        "rmse": float(np.sqrt(np.mean(difference**2))) if difference.size else 0.0,
        "pearson": pearson(first, second),
    }


def model_eval_audit(model: Any) -> dict[str, Any]:
    import torch

    training_modules = [name or "<root>" for name, module in model.named_modules() if module.training]
    batch_norm = []
    dropout = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            batch_norm.append(
                {
                    "name": name,
                    "training": module.training,
                    "track_running_stats": module.track_running_stats,
                    "running_mean_present": module.running_mean is not None,
                    "running_var_present": module.running_var is not None,
                }
            )
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            dropout.append({"name": name, "training": module.training, "p": module.p})
    return {
        "model_training": model.training,
        "training_modules": training_modules,
        "batch_norm": batch_norm,
        "dropout": dropout,
        "all_batch_norm_eval": all(not item["training"] for item in batch_norm),
        "all_dropout_eval": all(not item["training"] for item in dropout),
    }


def backend_state() -> dict[str, Any]:
    import torch

    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


@contextmanager
def cuda_backend_profile(name: str) -> Iterator[dict[str, Any]]:
    import torch

    if name not in CUDA_PROFILES:
        raise ValueError(f"Unknown CUDA profile: {name}")
    previous = backend_state()
    profile = CUDA_PROFILES[name]
    try:
        torch.set_float32_matmul_precision("high" if profile["tf32"] else "highest")
        torch.backends.cuda.matmul.allow_tf32 = profile["tf32"]
        torch.backends.cudnn.allow_tf32 = profile["tf32"]
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = profile["deterministic"]
        torch.use_deterministic_algorithms(profile["deterministic"], warn_only=True)
        yield backend_state()
    finally:
        torch.set_float32_matmul_precision(previous["float32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = previous["cuda_matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = previous["cudnn_allow_tf32"]
        torch.backends.cudnn.benchmark = previous["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = previous["cudnn_deterministic"]
        torch.use_deterministic_algorithms(
            previous["deterministic_algorithms"],
            warn_only=previous["deterministic_warn_only"],
        )


def capture_forward_gradient(
    model: Any,
    target_layer: Any,
    observed_layers: dict[str, Any],
    images: Any,
    target_class: int,
) -> dict[str, Any]:
    """Capture one label-free pass without populating parameter ``.grad`` fields."""
    import torch

    captured: dict[str, Any] = {}
    hooks = []
    for name, module in observed_layers.items():
        hooks.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, key=name: captured.__setitem__(key, output)
            )
        )
    try:
        model.eval()
        with torch.enable_grad():
            logits = model.forward_logits(images)
            if "target" not in captured:
                raise RuntimeError("target layer was not captured")
            target_activation = captured["target"]
            gradient = torch.autograd.grad(
                logits[:, target_class].sum(),
                target_activation,
                retain_graph=False,
                create_graph=False,
            )[0]
    finally:
        for hook in hooks:
            hook.remove()
    result = {name: value.detach() for name, value in captured.items()}
    result["logits"] = logits.detach()
    result["gradient"] = gradient.detach()
    return result
