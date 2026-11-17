from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from torch import Tensor


@dataclass
class CAMBatch:
    maps: Tensor
    logits: Tensor
    predicted_classes: Tensor
    target_classes: Tensor
    valid: Tensor


class CAMExtractor(Protocol):
    """Stable plug-in point for EdgeCAM now and Finer-CAM in Experiment 2."""

    def extract(self, images: Tensor, target_classes: Tensor | int) -> CAMBatch: ...

    def close(self) -> None: ...
