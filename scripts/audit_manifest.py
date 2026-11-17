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
    for row in rows:
        if not row.mask_relpath:
            continue
        with Image.open(args.mask_root / row.mask_relpath) as source:
            mask = np.asarray(source.convert("L"))
        pixel_values.update(int(value) for value in np.unique(mask))
        foreground_fractions.append(float(np.mean(mask > 0)))
        if mask.shape != (row.height, row.width):
            issues.append(
                {"level": "error", "sample_id": row.sample_id, "message": "mask/image size mismatch"}
            )
        if not np.any(mask > 0) or np.all(mask > 0):
            issues.append(
                {"level": "warning", "sample_id": row.sample_id, "message": "empty or full mask"}
            )
    serialized = [issue.__dict__ if hasattr(issue, "__dict__") else issue for issue in issues]
    report = {
        "rows": len(rows),
        "split_counts": dict(collections.Counter(row.split for row in rows)),
        "label_counts": dict(collections.Counter(str(row.label) for row in rows)),
        "mask_unique_values_frequency": dict(sorted(pixel_values.items())),
        "mask_foreground_fraction_min": min(foreground_fractions, default=None),
        "mask_foreground_fraction_max": max(foreground_fractions, default=None),
        "issues": serialized,
    }
    print(json.dumps(report, indent=2))
    if any(item["level"] == "error" for item in serialized):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
