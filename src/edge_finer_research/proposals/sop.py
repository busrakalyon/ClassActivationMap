from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray


class MissingSOPArtifactsError(FileNotFoundError):
    """Raised instead of silently substituting an unverified proposal generator."""


class ProposalProvider(Protocol):
    def get(self, sample_id: str) -> NDArray[np.bool_]: ...


class NpzProposalProvider:
    """Read precomputed proposals only; this does not claim to recreate their generator."""

    def __init__(self, root: str | Path, *, expected_count: int = 50, key: str = "proposals") -> None:
        self.root = Path(root)
        self.expected_count = expected_count
        self.key = key

    def get(self, sample_id: str) -> NDArray[np.bool_]:
        path = self.root / f"{sample_id}.npz"
        if not path.is_file():
            raise MissingSOPArtifactsError(f"No precomputed SOP proposals: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if self.key not in archive:
                raise ValueError(f"{path} has no '{self.key}' array")
            proposals = np.asarray(archive[self.key])
        if proposals.ndim != 3 or proposals.shape[0] != self.expected_count:
            raise ValueError(
                f"Expected [{self.expected_count},H,W] proposals, found {proposals.shape} in {path}"
            )
        return proposals.astype(bool)


def _morphology(mask: NDArray[np.bool_], kernel_size: int, operation: str) -> NDArray[np.bool_]:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    padding = kernel_size // 2
    padded = np.pad(mask, padding, mode="constant", constant_values=(operation == "erode"))
    windows = sliding_window_view(padded, (kernel_size, kernel_size))
    return windows.max(axis=(-2, -1)) if operation == "dilate" else windows.min(axis=(-2, -1))


def proposal_edge(
    masks: NDArray[np.bool_], *, kernel_size: int = 3, iterations: int = 1
) -> NDArray[np.bool_]:
    if masks.ndim != 3:
        raise ValueError("proposals must have shape [N,H,W]")
    dilated, eroded = masks.copy(), masks.copy()
    for _ in range(iterations):
        dilated = np.stack([_morphology(mask, kernel_size, "dilate") for mask in dilated])
        eroded = np.stack([_morphology(mask, kernel_size, "erode") for mask in eroded])
    return np.logical_and(dilated, np.logical_not(eroded))


@dataclass(frozen=True)
class SOPResult:
    mask: NDArray[np.bool_]
    proposal_index: int
    score: float
    scores: NDArray[np.float64]


def select_object_proposal(
    localization_edge_map: NDArray[np.floating],
    proposals: NDArray[np.bool_],
    *,
    expected_count: int = 50,
    kernel_size: int = 3,
    iterations: int = 1,
) -> SOPResult:
    """Implement paper Eq. 7--8/Algorithm 1 after proposals already exist.

    The unpublished/absent proposal-generation stage is deliberately outside
    this function. Input masks must therefore carry separate provenance.
    """
    heatmap = np.nan_to_num(np.asarray(localization_edge_map, dtype=np.float64))
    masks = np.asarray(proposals, dtype=bool)
    if masks.ndim != 3 or masks.shape[0] != expected_count:
        raise ValueError(f"Expected exactly {expected_count} proposal masks")
    if masks.shape[1:] != heatmap.shape:
        raise ValueError("proposal and localization map spatial shapes differ")
    edges = proposal_edge(masks, kernel_size=kernel_size, iterations=iterations)
    scores = np.sum(edges * heatmap[None, ...], axis=(1, 2), dtype=np.float64)
    index = int(np.argmax(scores))
    return SOPResult(masks[index], index, float(scores[index]), scores)
