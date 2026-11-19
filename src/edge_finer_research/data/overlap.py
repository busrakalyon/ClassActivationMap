from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .casia import parse_source_ids
from .manifest import ManifestRow, sha256_file


@dataclass(frozen=True)
class ImageFingerprint:
    row: ManifestRow
    file_sha256: str
    rgb_sha256: str
    perceptual_hash: int


def decoded_rgb_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"RGB:{rgb.width}x{rgb.height}:".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _dct_matrix(size: int) -> np.ndarray:
    coordinates = np.arange(size, dtype=np.float64)
    frequencies = coordinates[:, None]
    matrix = np.cos(np.pi * (2.0 * coordinates + 1.0) * frequencies / (2.0 * size))
    matrix[0] *= np.sqrt(1.0 / size)
    matrix[1:] *= np.sqrt(2.0 / size)
    return matrix


_PHASH_DCT = _dct_matrix(32)


def perceptual_hash(image: Image.Image) -> int:
    """Return a deterministic 64-bit DCT perceptual hash.

    This is only a candidate generator. A small Hamming distance does not prove
    common provenance and must be reviewed before excluding a test image.
    """
    grayscale = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.float64)
    coefficients = _PHASH_DCT @ pixels @ _PHASH_DCT.T
    low_frequency = coefficients[:8, :8]
    median = float(np.median(low_frequency.ravel()[1:]))
    bits = low_frequency >= median
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def fingerprint_image(row: ManifestRow, image_root: str | Path) -> ImageFingerprint:
    path = Path(image_root) / row.image_relpath
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as source:
        source.load()
        rgb_hash = decoded_rgb_sha256(source)
        phash = perceptual_hash(source)
    return ImageFingerprint(row, sha256_file(path), rgb_hash, phash)


def source_ids(row: ManifestRow) -> tuple[str, ...]:
    values = set(filter(None, row.source_ids.split(";")))
    values.update(parse_source_ids(row.image_relpath))
    return tuple(sorted(values))


def exact_cross_matches(
    left: Iterable[ImageFingerprint],
    right: Iterable[ImageFingerprint],
    *,
    attribute: str,
) -> list[tuple[ImageFingerprint, ImageFingerprint]]:
    index: dict[str, list[ImageFingerprint]] = {}
    for item in left:
        index.setdefault(str(getattr(item, attribute)), []).append(item)
    matches: list[tuple[ImageFingerprint, ImageFingerprint]] = []
    for item in right:
        for candidate in index.get(str(getattr(item, attribute)), []):
            matches.append((candidate, item))
    return matches


def perceptual_cross_matches(
    left: Iterable[ImageFingerprint],
    right: Iterable[ImageFingerprint],
    *,
    max_distance: int,
) -> list[tuple[ImageFingerprint, ImageFingerprint, int]]:
    if not 0 <= max_distance <= 64:
        raise ValueError("max_distance must be between 0 and 64")
    left_items = list(left)
    matches: list[tuple[ImageFingerprint, ImageFingerprint, int]] = []
    for test_item in right:
        for candidate in left_items:
            distance = (candidate.perceptual_hash ^ test_item.perceptual_hash).bit_count()
            if distance <= max_distance:
                matches.append((candidate, test_item, distance))
    return matches


def source_cross_matches(
    left: Iterable[ImageFingerprint],
    right: Iterable[ImageFingerprint],
) -> list[tuple[ImageFingerprint, ImageFingerprint, tuple[str, ...]]]:
    index: dict[str, list[ImageFingerprint]] = {}
    for item in left:
        for source_id in source_ids(item.row):
            index.setdefault(source_id, []).append(item)
    pairs: dict[tuple[str, str], tuple[ImageFingerprint, ImageFingerprint, set[str]]] = {}
    for test_item in right:
        candidates: dict[str, tuple[ImageFingerprint, set[str]]] = {}
        for source_id in source_ids(test_item.row):
            for candidate in index.get(source_id, []):
                current = candidates.setdefault(candidate.row.sample_id, (candidate, set()))
                current[1].add(source_id)
        for candidate, shared in candidates.values():
            pairs[(candidate.row.sample_id, test_item.row.sample_id)] = (
                candidate,
                test_item,
                shared,
            )
    return [
        (left_item, right_item, tuple(sorted(shared)))
        for left_item, right_item, shared in pairs.values()
    ]
