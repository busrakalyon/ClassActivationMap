from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .arcface import ArcMarginProduct


class EfficientNetArcFace(nn.Module):
    """EfficientNet-B0 classifier matching the supplied EdgeCAM code where documented.

    ``legacy_imagenet_logits`` preserves the downloaded repository's unusual
    1000-logit -> 512 embedding bridge. ``pooled_features`` is provided only as
    an explicit ablation and is not the Experiment-1 default.
    """

    def __init__(
        self,
        *,
        num_classes: int = 2,
        embedding_dim: int = 512,
        dropout_before_embedding: float = 0.3,
        dropout_after_embedding: float = 0.2,
        arc_scale: float = 50.0,
        arc_margin: float = 0.5,
        arc_easy_margin: bool = False,
        feature_mode: str = "legacy_imagenet_logits",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        self.feature_mode = feature_mode
        if feature_mode == "legacy_imagenet_logits":
            input_dim = 1000
        elif feature_mode == "pooled_features":
            input_dim = self.backbone.classifier[1].in_features
        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")
        self.dropout_before_embedding = nn.Dropout(dropout_before_embedding)
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.dropout_after_embedding = nn.Dropout(dropout_after_embedding)
        self.arcface = ArcMarginProduct(
            embedding_dim, num_classes, arc_scale, arc_margin, arc_easy_margin
        )

    def extract_embedding(self, images: Tensor) -> Tensor:
        if self.feature_mode == "legacy_imagenet_logits":
            features = self.backbone(images)
        else:
            features = self.backbone.features(images)
            features = self.backbone.avgpool(features)
            features = torch.flatten(features, 1)
        features = self.dropout_before_embedding(features)
        return self.dropout_after_embedding(self.embedding(features))

    def forward_train(self, images: Tensor, labels: Tensor) -> Tensor:
        return self.arcface.forward_train(self.extract_embedding(images), labels)

    def forward_logits(self, images: Tensor) -> Tensor:
        """Label-free logits used by all validation, test and CAM code."""
        return self.arcface.forward_logits(self.extract_embedding(images))

    def forward(self, images: Tensor, labels: Tensor | None = None) -> Tensor:
        return self.forward_logits(images) if labels is None else self.forward_train(images, labels)

    def resolve_target_layer(self, dotted_path: str) -> nn.Module:
        module: nn.Module = self
        for token in dotted_path.split("."):
            module = module[int(token)] if token.isdigit() else getattr(module, token)
        return module
