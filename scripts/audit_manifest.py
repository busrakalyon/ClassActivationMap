from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from PIL import Image

from edge_finer_research.data.manifest import read_manifest, validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--require-binary-masks", action="store_true")
    parser.add_argument("--require-mixed-masks", action="store_true")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-split")
    parser.add_argument("--expected-label", type=int, choices=(0, 1))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    issues = validate_manifest(
        rows,
        image_root=args.image_root,
        mask_root=args.mask_root,
        verify_hashes=args.verify_hashes,
    )
    pixel_values: collections.Counter[int] = collections.Counter()
    foreground_fractions: list[float] = []
    binary_mask_count = 0
    mixed_mask_count = 0
    for row in rows:
        if not row.mask_relpath:
            continue
        with Image.open(args.mask_root / row.mask_relpath) as source:
            mask = np.asarray(source.convert("L"))
        pixel_values.update(int(value) for value in np.unique(mask))
        foreground_fractions.append(float(np.mean(mask > 0)))
        unique_values = set(int(value) for value in np.unique(mask))
        is_binary = unique_values.issubset({0, 255})
        is_mixed = 0 in unique_values and 255 in unique_values
        binary_mask_count += int(is_binary)
        mixed_mask_count += int(is_mixed)
        if mask.shape != (row.height, row.width):
            issues.append(
                {
                    "level": "error",
                    "sample_id": row.sample_id,
                    "message": "mask/image size mismatch",
                }
            )
        if args.require_binary_masks and not is_binary:
            issues.append(
                {
                    "level": "error",
                    "sample_id": row.sample_id,
                    "message": f"mask is not binary 0/255: {sorted(unique_values)}",
                }
            )
        if not np.any(mask > 0) or np.all(mask > 0):
            issues.append(
                {
                    "level": "error" if args.require_mixed_masks else "warning",
                    "sample_id": row.sample_id,
                    "message": "empty or full mask",
                }
            )
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        issues.append(
            {
                "level": "error",
                "sample_id": "__manifest__",
                "message": f"expected {args.expected_rows} rows, found {len(rows)}",
            }
        )
    if args.expected_split is not None and any(row.split != args.expected_split for row in rows):
        issues.append(
            {
                "level": "error",
                "sample_id": "__manifest__",
                "message": f"not every row has split={args.expected_split}",
            }
        )
    if args.expected_label is not None and any(row.label != args.expected_label for row in rows):
        issues.append(
            {
                "level": "error",
                "sample_id": "__manifest__",
                "message": f"not every row has label={args.expected_label}",
            }
        )
    serialized = [issue.__dict__ if hasattr(issue, "__dict__") else issue for issue in issues]
    error_count = sum(issue["level"] == "error" for issue in serialized)
    report = {
        "rows": len(rows),
        "split_counts": dict(collections.Counter(row.split for row in rows)),
        "label_counts": dict(collections.Counter(str(row.label) for row in rows)),
        "mask_unique_values_frequency": dict(sorted(pixel_values.items())),
        "mask_count": len(foreground_fractions),
        "binary_mask_count": binary_mask_count,
        "mixed_mask_count": mixed_mask_count,
        "mask_foreground_fraction_min": min(foreground_fractions, default=None),
        "mask_foreground_fraction_max": max(foreground_fractions, default=None),
        "error_count": error_count,
        "issues": serialized,
        "issues": serialized,
    }
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if error_count:
        raise RuntimeError(f"Manifest audit found {error_count} errors")
    if any(item["level"] == "error" for item in serialized):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
