from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import torch
    from PIL import Image, ImageFilter
    from torch.utils.data import Dataset
    from torchvision import transforms
except ImportError as exc:  # pragma: no cover - exercised on the Ubuntu runtime
    raise ImportError("Dataset runtime requires torch, torchvision and Pillow") from exc

from .manifest import ManifestRow


class RandomJPEGCompression:
    def __init__(self, probability: float, quality_min: int, quality_max: int) -> None:
        self.probability = probability
        self.quality_min = quality_min
        self.quality_max = quality_max

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(self.quality_min, self.quality_max))
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")


class RandomGaussianBlur:
    def __init__(self, probability: float, kernel_size: int) -> None:
        self.probability = probability
        self.radius = max(0.1, (kernel_size - 1) / 6.0)

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(self.radius)) if random.random() < self.probability else image


def build_transforms(config: dict, *, train: bool) -> Callable:
    image_size = int(config["experiment"]["image_size"])
    operations: list[Callable] = []
    if train:
        augmentation = config["augmentation"]
        operations.extend(
            [
                transforms.RandomHorizontalFlip(float(augmentation["horizontal_flip_probability"])),
                RandomJPEGCompression(
                    float(augmentation["jpeg_probability"]),
                    int(augmentation["jpeg_quality_min"]),
                    int(augmentation["jpeg_quality_max"]),
                ),
                RandomGaussianBlur(
                    float(augmentation["gaussian_blur_probability"]),
                    int(augmentation["gaussian_blur_kernel"]),
                ),
            ]
        )
    operations.extend(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(operations)


class ManifestClassificationDataset(Dataset):
    def __init__(self, rows: list[ManifestRow], image_root: str | Path, transform: Callable) -> None:
        self.rows = rows
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.image_root / row.image_relpath) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.tensor(row.label, dtype=torch.long), row.sample_id


class ManifestLocalizationDataset(ManifestClassificationDataset):
    def __init__(
        self,
        rows: list[ManifestRow],
        image_root: str | Path,
        mask_root: str | Path,
        transform: Callable,
        *,
        output_size: tuple[int, int],
    ) -> None:
        super().__init__(rows, image_root, transform)
        self.mask_root = Path(mask_root)
        self.output_size = output_size

    def __getitem__(self, index: int):
        image, label, sample_id = super().__getitem__(index)
        row = self.rows[index]
        if row.mask_relpath:
            with Image.open(self.mask_root / row.mask_relpath) as mask_image:
                mask_image = mask_image.convert("L").resize(self.output_size[::-1], Image.Resampling.NEAREST)
                mask = torch.from_numpy(np.asarray(mask_image).copy() > 0)
        else:
            mask = torch.zeros(self.output_size, dtype=torch.bool)
        return image, label, mask, sample_id

