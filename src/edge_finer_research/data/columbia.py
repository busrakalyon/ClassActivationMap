from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np
from PIL import Image


TIFF_EXTENSIONS = {".tif", ".tiff"}


def filename_from_legacy_entry(entry: str) -> str:
    value = entry.strip().strip('"').strip("'")
    if not value:
        raise ValueError("Empty vallist entry")
    filename = value.replace("\\", "/").rsplit("/", 1)[-1]
    if not filename or Path(filename).suffix.lower() not in TIFF_EXTENSIONS:
        raise ValueError(f"vallist entry is not a TIFF path: {entry!r}")
    return filename


def read_unique_vallist(path: str | Path, *, expected_count: int) -> list[str]:
    entries = [
        filename_from_legacy_entry(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(entries) != expected_count:
        raise ValueError(f"Expected {expected_count} vallist rows, found {len(entries)}")
    folded = [entry.casefold() for entry in entries]
    if len(set(folded)) != len(folded):
        raise ValueError("vallist contains duplicate filenames")
    return entries


def threshold_mask_image(source_path: str | Path) -> tuple[np.ndarray, set[int]]:
    with Image.open(source_path) as source:
        grayscale = np.asarray(source.convert("L"), dtype=np.uint8)
    binary = np.where(grayscale >= 128, 255, 0).astype(np.uint8)
    return binary, set(int(value) for value in np.unique(binary))


def write_binary_png_atomic(path: str | Path, binary_mask: np.ndarray) -> None:
    destination = Path(path)
    values = np.asarray(binary_mask)
    if values.ndim != 2 or values.dtype != np.uint8:
        raise ValueError("Binary mask must be a two-dimensional uint8 array")
    if not set(int(value) for value in np.unique(values)).issubset({0, 255}):
        raise ValueError("Binary mask contains values other than 0 and 255")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.tmp.png")
    Image.fromarray(values, mode="L").save(temporary, format="PNG")
    os.replace(temporary, destination)


def verify_existing_binary_png(path: str | Path, expected: np.ndarray) -> None:
    with Image.open(path) as source:
        actual = np.asarray(source.convert("L"), dtype=np.uint8)
    if not np.array_equal(actual, expected):
        raise ValueError(f"Existing derived mask differs from thresholded source: {path}")
