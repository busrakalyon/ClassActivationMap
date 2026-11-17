from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ArcMarginProduct(nn.Module):
    """ArcFace head with separate label-aware training and label-free inference APIs."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: float = 50.0,
        margin: float = 0.5,
        easy_margin: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = float(scale)
        self.margin = float(margin)
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.threshold = math.cos(math.pi - self.margin)
        self.margin_correction = math.sin(math.pi - self.margin) * self.margin

    def cosine(self, embeddings: Tensor) -> Tensor:
        return F.linear(F.normalize(embeddings), F.normalize(self.weight))

    def forward_logits(self, embeddings: Tensor) -> Tensor:
        """Return scaled cosine logits without labels; use this for validation/CAM/test."""
        return self.cosine(embeddings) * self.scale

    def forward_train(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        cosine = self.cosine(embeddings)
        sine = torch.sqrt(torch.clamp(1.0 - cosine.square(), min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.threshold, phi, cosine - self.margin_correction)
        one_hot = F.one_hot(labels, num_classes=self.out_features).to(dtype=cosine.dtype)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale

    def forward(self, embeddings: Tensor, labels: Tensor | None = None) -> Tensor:
        return self.forward_logits(embeddings) if labels is None else self.forward_train(embeddings, labels)
