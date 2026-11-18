from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from edge_finer_research.cams.edgecam import PaperEdgeCAM
from edge_finer_research.cams.epoch_fusion import fuse_epoch_cams


class SpatialMaskLayer(nn.Module):
    def __init__(self, mask: Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, images: Tensor) -> Tensor:
        return images * self.mask


class DeterministicFakeModel(nn.Module):
    """Tiny control model whose fake logit positively depends on a known activation."""

    def __init__(self, spatial_mask: Tensor) -> None:
        super().__init__()
        self.target_layer = SpatialMaskLayer(spatial_mask)

    def forward_logits(self, images: Tensor) -> Tensor:
        activation = self.target_layer(images)
        fake_score = activation.mean(dim=(1, 2, 3))
        return torch.stack((-fake_score, fake_score), dim=1)


def run_deterministic_cam_control(device: str | torch.device = "cpu") -> dict[str, object]:
    """Prove that valid, non-zero CAMs enter max fusion.

    This is an infrastructure control, not a classifier or research experiment.
    """
    target_device = torch.device(device)
    image = torch.linspace(0.1, 1.0, steps=16, device=target_device).view(1, 1, 4, 4)
    image.requires_grad_(True)
    left = torch.zeros((1, 1, 4, 4), device=target_device)
    right = torch.zeros((1, 1, 4, 4), device=target_device)
    left[..., :2] = 1.0
    right[..., 2:] = 1.0

    maps: list[np.ndarray] = []
    valid: list[bool] = []
    for spatial_mask in (left, right):
        model = DeterministicFakeModel(spatial_mask).to(target_device)
        with PaperEdgeCAM(
            model,
            model.target_layer,
            gate_on_predicted_class=True,
            output_size=(4, 4),
        ) as extractor:
            result = extractor.extract(image, 1)
        maps.append(result.maps[0].cpu().numpy())
        valid.append(bool(result.valid[0].item()))

    nonzero = [bool(np.any(cam > 0.0)) for cam in maps]
    expected_max = np.maximum(maps[0], maps[1])
    fusion = fuse_epoch_cams(maps, valid=valid, method="max")
    fusion_matches_max = bool(np.allclose(fusion.map, expected_max))
    both_maps_contribute = bool(np.any(maps[0] > maps[1]) and np.any(maps[1] > maps[0]))
    checks = {
        "at_least_one_valid_fake_cam": any(valid),
        "all_generated_cams_nonzero": all(nonzero),
        "valid_maps_included_in_max_fusion": (
            fusion.valid_count == sum(valid) == 2 and fusion_matches_max and both_maps_contribute
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"Deterministic CAM control failed: {checks}")
    return {
        **checks,
        "valid_flags": valid,
        "nonzero_flags": nonzero,
        "valid_count": fusion.valid_count,
        "total_count": fusion.total_count,
        "fusion_matches_elementwise_max": fusion_matches_max,
        "both_epoch_maps_contribute": both_maps_contribute,
        "note": "Infrastructure control only; not a research result.",
    }
