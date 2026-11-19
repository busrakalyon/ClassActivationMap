from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .sop import MissingSOPArtifactsError


@dataclass(frozen=True)
class MCGProposalCache:
    masks_hwk: NDArray[np.uint8]
    scores: NDArray[np.float64]
    ranks: NDArray[np.int32]

    @property
    def masks_nhw(self) -> NDArray[np.bool_]:
        return np.moveaxis(self.masks_hwk, 2, 0).astype(bool, copy=False)


def _validate_arrays(
    masks: np.ndarray,
    scores: np.ndarray,
    ranks: np.ndarray,
    *,
    expected_height: int,
    expected_width: int,
    expected_count: int,
) -> MCGProposalCache:
    if masks.shape != (expected_height, expected_width, expected_count):
        raise ValueError(
            "MCG masks must have shape "
            f"[{expected_height},{expected_width},{expected_count}], found {masks.shape}"
        )
    if masks.dtype not in (np.dtype(np.bool_), np.dtype(np.uint8)):
        raise TypeError(f"MCG masks must be bool or uint8, found {masks.dtype}")
    if not np.logical_or(masks == 0, masks == 1).all():
        raise ValueError("MCG masks contain values other than 0 and 1")
    if scores.shape != (expected_count,) or not np.isfinite(scores).all():
        raise ValueError(f"MCG scores must be finite [{expected_count}], found {scores.shape}")
    expected_ranks = np.arange(1, expected_count + 1, dtype=np.int32)
    if ranks.shape != (expected_count,) or not np.array_equal(ranks, expected_ranks):
        raise ValueError("MCG ranks must be the one-based sequence 1..top_k")
    return MCGProposalCache(
        masks.astype(np.uint8, copy=False),
        scores.astype(np.float64, copy=False),
        ranks.astype(np.int32, copy=False),
    )


def load_mcg_cache(
    path: str | Path,
    *,
    expected_height: int,
    expected_width: int,
    expected_count: int = 50,
) -> MCGProposalCache:
    source = Path(path)
    if not source.is_file():
        raise MissingSOPArtifactsError(f"No MCG proposal cache: {source}")
    with np.load(source, allow_pickle=False) as archive:
        required = {"masks", "scores", "ranks"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{source} is missing NPZ keys: {sorted(missing)}")
        masks = np.asarray(archive["masks"])
        scores = np.asarray(archive["scores"])
        ranks = np.asarray(archive["ranks"])
    return _validate_arrays(
        masks,
        scores,
        ranks,
        expected_height=expected_height,
        expected_width=expected_width,
        expected_count=expected_count,
    )


def parse_matlab_intermediate(
    directory: str | Path,
    *,
    expected_height: int,
    expected_width: int,
    expected_count: int = 50,
) -> tuple[MCGProposalCache, dict[str, Any]]:
    root = Path(directory)
    completed = root / "completed.json"
    metadata_path = root / "metadata.json"
    if not completed.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Incomplete MATLAB MCG output: {root}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata["height"]) != expected_height or int(metadata["width"]) != expected_width:
        raise ValueError("MATLAB MCG output dimensions differ from the manifest")
    if int(metadata["saved_top_k"]) != expected_count:
        raise ValueError("MATLAB MCG output top-k differs from the requested count")
    if int(metadata["proposal_count_total"]) < expected_count:
        raise ValueError("MATLAB MCG produced fewer proposals than requested")

    mask_path = root / "masks_uint8_f_order.bin"
    score_path = root / "scores_float64_le.bin"
    expected_mask_values = expected_height * expected_width * expected_count
    masks_flat = np.fromfile(mask_path, dtype=np.uint8)
    if masks_flat.size != expected_mask_values:
        raise ValueError(
            f"MATLAB mask payload has {masks_flat.size} values, expected {expected_mask_values}"
        )
    masks = masks_flat.reshape(
        (expected_height, expected_width, expected_count), order="F"
    )
    scores = np.fromfile(score_path, dtype="<f8")
    ranks = np.arange(1, expected_count + 1, dtype=np.int32)
    cache = _validate_arrays(
        masks,
        scores,
        ranks,
        expected_height=expected_height,
        expected_width=expected_width,
        expected_count=expected_count,
    )
    return cache, metadata


def write_mcg_cache_atomic(path: str | Path, cache: MCGProposalCache) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        np.savez_compressed(
            handle,
            masks=cache.masks_hwk,
            scores=cache.scores,
            ranks=cache.ranks,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def validate_reusable_cache(
    cache_path: str | Path,
    metadata_path: str | Path,
    *,
    sample_id: str,
    image_sha256: str,
    expected_height: int,
    expected_width: int,
    expected_count: int = 50,
    mode: str = "accurate",
    mcg_source_commit: str | None = None,
) -> bool:
    cache_source, metadata_source = Path(cache_path), Path(metadata_path)
    if not cache_source.exists() and not metadata_source.exists():
        return False
    if not cache_source.is_file() or not metadata_source.is_file():
        raise ValueError(f"Partial MCG cache exists for {sample_id}")
    load_mcg_cache(
        cache_source,
        expected_height=expected_height,
        expected_width=expected_width,
        expected_count=expected_count,
    )
    metadata = json.loads(metadata_source.read_text(encoding="utf-8"))
    expected = {
        "generator": "official_mcg_reconstructed",
        "sample_id": sample_id,
        "image_sha256": image_sha256,
        "image_height": expected_height,
        "image_width": expected_width,
        "saved_top_k": expected_count,
        "mcg_mode": mode,
    }
    if mcg_source_commit is not None:
        expected["mcg_source_commit"] = mcg_source_commit
    differences = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if differences:
        raise ValueError(f"MCG cache metadata mismatch for {sample_id}: {differences}")
    if int(metadata.get("proposal_count_total", 0)) < expected_count:
        raise ValueError(f"MCG proposal_count_total is invalid for {sample_id}")
    return True


def mcg_git_commit(mcg_root: str | Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(Path(mcg_root)), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


class MCGNpzProposalProvider:
    def __init__(self, root: str | Path, *, expected_count: int = 50) -> None:
        self.root = Path(root)
        self.expected_count = expected_count

    def get_for_shape(
        self, sample_id: str, *, height: int, width: int
    ) -> NDArray[np.bool_]:
        cache = load_mcg_cache(
            self.root / f"{sample_id}.npz",
            expected_height=height,
            expected_width=width,
            expected_count=self.expected_count,
        )
        return cache.masks_nhw
