from __future__ import annotations

from typing import Any


PROFILE_NAME = "scientific_cuda_fp32_no_tf32_batch1"


def _autocast_enabled(torch: Any) -> bool:
    try:
        return bool(torch.is_autocast_enabled("cuda"))
    except TypeError:  # Older supported PyTorch API.
        return bool(torch.is_autocast_enabled())


def backend_state() -> dict[str, Any]:
    import torch

    return {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cuda_autocast_enabled": _autocast_enabled(torch),
    }


def apply_scientific_cam_profile(device: Any, *, batch_size: int) -> dict[str, Any]:
    """Apply and verify the fixed scientific CAM inference profile."""
    import torch

    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("The scientific CAM profile requires a CUDA device")
    if batch_size != 1:
        raise ValueError("The scientific CAM profile requires batch_size=1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA profile requested but torch.cuda.is_available() is false")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    actual = backend_state()
    if actual["cuda_matmul_allow_tf32"] or actual["cudnn_allow_tf32"]:
        raise RuntimeError(f"TF32 disable request did not take effect: {actual}")
    if actual["float32_matmul_precision"] != "highest":
        raise RuntimeError(f"FP32 matmul precision request did not take effect: {actual}")
    if actual["cuda_autocast_enabled"]:
        raise RuntimeError("CUDA autocast is unexpectedly active")
    return {
        "profile_name": PROFILE_NAME,
        "batch_size": 1,
        "dtype": "float32",
        "amp": False,
        "device": str(resolved),
        "actual_backend": actual,
        "settings_verified": True,
    }


def verify_scientific_cam_runtime(model: Any, images: Any) -> dict[str, Any]:
    """Fail closed if model, input, or backend drifts from the applied profile."""
    import torch

    actual = backend_state()
    if model.training or any(module.training for module in model.modules()):
        raise RuntimeError("CAM model or one of its modules is not in eval mode")
    parameter_dtypes = {parameter.dtype for parameter in model.parameters()}
    if parameter_dtypes != {torch.float32}:
        raise RuntimeError(f"CAM model parameters are not uniformly float32: {parameter_dtypes}")
    if images.dtype != torch.float32:
        raise RuntimeError(f"CAM input dtype is {images.dtype}, expected torch.float32")
    if images.shape[0] != 1:
        raise RuntimeError(f"CAM runtime batch is {images.shape[0]}, expected one")
    if actual["cuda_matmul_allow_tf32"] or actual["cudnn_allow_tf32"]:
        raise RuntimeError(f"TF32 became enabled during CAM extraction: {actual}")
    if actual["float32_matmul_precision"] != "highest":
        raise RuntimeError(f"FP32 matmul precision drifted during CAM extraction: {actual}")
    if actual["cuda_autocast_enabled"]:
        raise RuntimeError("CUDA autocast became active during CAM extraction")
    return {
        "model_eval_verified": True,
        "parameter_dtype": "float32",
        "input_dtype": "float32",
        "runtime_batch_size": 1,
        "amp_verified_disabled": True,
        "actual_backend": actual,
    }
