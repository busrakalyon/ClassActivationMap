from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


MANIFEST_FIELDS = (
    "sample_id",
    "dataset",
    "split",
    "group_id",
    "image_relpath",
    "mask_relpath",
    "label",
    "manipulation_type",
    "source_ids",
    "image_sha256",
    "mask_sha256",
    "width",
    "height",
    "source_revision",
)


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    dataset: str
    split: str
    group_id: str
    image_relpath: str
    mask_relpath: str
    label: int
    manipulation_type: str = "unknown"
    source_ids: str = ""
    image_sha256: str = ""
    mask_sha256: str = ""
    width: int = 0
    height: int = 0
    source_revision: str = ""

    @classmethod
    def from_dict(cls, item: dict[str, str]) -> "ManifestRow":
        missing = [field for field in MANIFEST_FIELDS if field not in item]
        if missing:
            raise ValueError(f"Manifest columns missing: {missing}")
        values = dict(item)
        values["label"] = int(values["label"])
        values["width"] = int(values["width"] or 0)
        values["height"] = int(values["height"] or 0)
        return cls(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    sample_id: str
    message: str


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: str | Path) -> list[ManifestRow]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [ManifestRow.from_dict(item) for item in csv.DictReader(handle)]


def write_manifest(rows: Iterable[ManifestRow], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def validate_manifest(
    rows: Iterable[ManifestRow],
    *,
    image_root: str | Path | None = None,
    mask_root: str | Path | None = None,
    verify_hashes: bool = False,
) -> list[ValidationIssue]:
    """Validate labels, IDs, group isolation and optionally files/hashes."""
    rows = list(rows)
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()
    group_splits: dict[str, set[str]] = {}
    image_root_path = Path(image_root) if image_root is not None else None
    mask_root_path = Path(mask_root) if mask_root is not None else image_root_path

    for row in rows:
        if row.sample_id in seen_ids:
            issues.append(ValidationIssue("error", row.sample_id, "duplicate sample_id"))
        seen_ids.add(row.sample_id)
        if row.label not in (0, 1):
            issues.append(ValidationIssue("error", row.sample_id, "label must be 0 or 1"))
        if not row.group_id:
            issues.append(ValidationIssue("error", row.sample_id, "group_id is empty"))
        group_splits.setdefault(row.group_id, set()).add(row.split)
        if row.label == 1 and not row.mask_relpath:
            issues.append(ValidationIssue("error", row.sample_id, "tampered sample has no mask"))

        if image_root_path is not None:
            image_path = image_root_path / row.image_relpath
            if not image_path.is_file():
                issues.append(ValidationIssue("error", row.sample_id, f"missing image: {image_path}"))
            elif verify_hashes and row.image_sha256 and sha256_file(image_path) != row.image_sha256:
                issues.append(ValidationIssue("error", row.sample_id, "image SHA-256 mismatch"))

        if row.mask_relpath and mask_root_path is not None:
            mask_path = mask_root_path / row.mask_relpath
            if not mask_path.is_file():
                issues.append(ValidationIssue("error", row.sample_id, f"missing mask: {mask_path}"))
            elif verify_hashes and row.mask_sha256 and sha256_file(mask_path) != row.mask_sha256:
                issues.append(ValidationIssue("error", row.sample_id, "mask SHA-256 mismatch"))

    for group_id, splits in sorted(group_splits.items()):
        if len(splits) > 1:
            issues.append(
                ValidationIssue("error", group_id, f"group leaks across splits: {sorted(splits)}")
            )
    return issues

