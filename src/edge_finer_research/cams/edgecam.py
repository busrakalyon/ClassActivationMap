from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .base import CAMBatch


def paper_edgecam_tensor(
    activations: Tensor,
    gradients: Tensor,
    *,
    epsilon: float = 1e-8,
    apply_relu: bool = True,
    normalize: bool = True,
) -> Tensor:
    if activations.shape != gradients.shape or activations.ndim != 4:
        raise ValueError("activations and gradients must have equal [B,C,H,W] shapes")
    gradient_squared = gradients.square()
    denominator = 2.0 * gradient_squared + activations.sum(dim=(-2, -1), keepdim=True) * gradients.pow(3)
    alpha = torch.where(
        denominator.abs() > epsilon,
        gradient_squared / denominator,
        torch.zeros_like(gradient_squared),
    )
    weights = alpha * gradients.clamp_min(0.0)
    maps = (weights * activations).sum(dim=1)
    if apply_relu:
        maps = maps.relu()
    maps = torch.nan_to_num(maps)
    if normalize:
        flat = maps.flatten(1)
        minimum = flat.amin(dim=1).view(-1, 1, 1)
        span = (flat.amax(dim=1) - flat.amin(dim=1)).view(-1, 1, 1)
        maps = torch.where(span > epsilon, (maps - minimum) / span, torch.zeros_like(maps))
    return maps


class PaperEdgeCAM:
    """Hook-based reimplementation of the paper equations, not an author release."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        *,
        epsilon: float = 1e-8,
        apply_relu: bool = True,
        normalize: bool = True,
        gate_on_predicted_class: bool = False,
        output_size: tuple[int, int] | None = None,
    ) -> None:
        self.model = model
        self.target_layer = target_layer
        self.epsilon = epsilon
        self.apply_relu = apply_relu
        self.normalize = normalize
        self.gate_on_predicted_class = gate_on_predicted_class
        self.output_size = output_size
        self._activation: Tensor | None = None
        self._hook = target_layer.register_forward_hook(self._capture_activation)

    def _capture_activation(self, _module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        self._activation = output

    def extract(self, images: Tensor, target_classes: Tensor | int) -> CAMBatch:
        self.model.eval()
        self._activation = None
        with torch.enable_grad():
            logits = self.model.forward_logits(images)
            activation = self._activation
            if activation is None:
                raise RuntimeError("Target layer did not run during forward pass")
            if isinstance(target_classes, int):
                targets = torch.full((images.shape[0],), target_classes, device=images.device, dtype=torch.long)
            else:
                targets = target_classes.to(device=images.device, dtype=torch.long)
            selected = logits.gather(1, targets[:, None]).sum()
            gradients = torch.autograd.grad(selected, activation, retain_graph=False, create_graph=False)[0]
            maps = paper_edgecam_tensor(
                activation,
                gradients,
                epsilon=self.epsilon,
                apply_relu=self.apply_relu,
                normalize=self.normalize,
            )
        if self.output_size is not None and maps.shape[-2:] != self.output_size:
            maps = F.interpolate(maps[:, None], self.output_size, mode="bilinear", align_corners=False)[:, 0]
        predicted = logits.argmax(dim=1)
        valid = predicted.eq(targets) if self.gate_on_predicted_class else torch.ones_like(predicted, dtype=torch.bool)
        if self.gate_on_predicted_class:
            maps = maps * valid[:, None, None]
        return CAMBatch(maps.detach(), logits.detach(), predicted.detach(), targets.detach(), valid.detach())

    def close(self) -> None:
        self._hook.remove()

    def __enter__(self) -> "PaperEdgeCAM":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
