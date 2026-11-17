from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from .manifest import ManifestRow


SOURCE_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9])([a-z]{3})_?(\d{4,5})(?!\d)", re.IGNORECASE)


def parse_source_ids(filename: str) -> tuple[str, ...]:
    """Extract CASIA host/donor IDs such as cha00063 from a filename."""
    stem = Path(filename).stem
    values = {f"{prefix.lower()}{digits}" for prefix, digits in SOURCE_ID_PATTERN.findall(stem)}
    return tuple(sorted(values))


def infer_manipulation_type(filename: str) -> str:
    stem = Path(filename).stem.lower()
    if stem.startswith(("tp_s_", "sp_s_")):
        return "copy_move"
    if stem.startswith(("tp_d_", "sp_d_")):
        return "splicing"
    return "unknown"


class DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def assign_connected_group_ids(rows: Iterable[ManifestRow]) -> list[ManifestRow]:
    """Keep authentic sources, donors, hosts and their derivatives in one component."""
    rows = list(rows)
    sets = DisjointSet()
    row_sources: dict[str, tuple[str, ...]] = {}
    for row in rows:
        source_ids = tuple(filter(None, row.source_ids.split(";"))) or parse_source_ids(row.image_relpath)
        if not source_ids:
            source_ids = (f"unparsed:{row.sample_id}",)
        row_sources[row.sample_id] = source_ids
        for source_id in source_ids[1:]:
            sets.union(source_ids[0], source_id)

    output: list[ManifestRow] = []
    for row in rows:
        group_id = sets.find(row_sources[row.sample_id][0])
        output.append(replace(row, group_id=group_id, source_ids=";".join(row_sources[row.sample_id])))
    return output


def deterministic_validation_split(
    rows: Iterable[ManifestRow], *, validation_fraction: float, seed: int
) -> list[ManifestRow]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    output: list[ManifestRow] = []
    split_by_group: dict[str, str] = {}
    for row in rows:
        if row.group_id not in split_by_group:
            token = hashlib.sha256(f"{seed}:{row.group_id}".encode()).digest()
            value = int.from_bytes(token[:8], "big") / float(2**64)
            split_by_group[row.group_id] = "val" if value < validation_fraction else "train"
        output.append(replace(row, split=split_by_group[row.group_id]))
    return output


def find_mask(image_path: Path, mask_root: Path) -> Path | None:
    """Find common third-party CASIA ground-truth names; ambiguity is rejected."""
    candidates: list[Path] = []
    for suffix in (".png", "_gt.png", "_gt0.png", "_gt1.png", ".bmp"):
        candidate = mask_root / f"{image_path.stem}{suffix}"
        if candidate.is_file():
            candidates.append(candidate)
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous masks for {image_path.name}: {candidates}")
    return candidates[0] if candidates else None

