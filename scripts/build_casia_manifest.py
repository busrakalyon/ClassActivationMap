from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from edge_finer_research.data.casia import (
    assign_connected_group_ids,
    deterministic_validation_split,
    infer_manipulation_type,
    parse_source_ids,
)
from edge_finer_research.data.manifest import ManifestRow, sha256_file, validate_manifest, write_manifest


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def files_under(path: Path) -> list[Path]:
    return sorted(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_EXTENSIONS)


def mask_index(mask_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in files_under(mask_root):
        stem = path.stem
        for ending in ("_gt0", "_gt1", "_gt"):
            if stem.lower().endswith(ending):
                stem = stem[: -len(ending)]
                break
        keys = {stem.lower(), Path(stem).stem.lower()}
        for key in keys:
            index.setdefault(key, []).append(path)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a provenance-rich CASIA CSV manifest")
    parser.add_argument("--dataset", required=True, choices=("casia_v1", "casia_v2"))
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authentic-dir", default="Au")
    parser.add_argument("--tampered-dir", default="Tp")
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--hashes", action="store_true")
    args = parser.parse_args()

    image_root, mask_root = args.image_root.resolve(), args.mask_root.resolve()
    indexed_masks = mask_index(mask_root)
    rows: list[ManifestRow] = []
    for label, relative_dir in ((0, args.authentic_dir), (1, args.tampered_dir)):
        for image_path in files_under(image_root / relative_dir):
            candidates = indexed_masks.get(image_path.stem.lower(), []) if label else []
            if label and len(candidates) != 1:
                raise RuntimeError(f"Expected one mask for {image_path}, found {candidates}")
            mask_path = candidates[0] if candidates else None
            with Image.open(image_path) as image:
                width, height = image.size
            source_ids = parse_source_ids(image_path.name)
            rows.append(
                ManifestRow(
                    sample_id=f"{args.dataset}__{relative_dir}__{image_path.stem}",
                    dataset=args.dataset,
                    split="test" if args.dataset == "casia_v1" else "train",
                    group_id="pending",
                    image_relpath=image_path.relative_to(image_root).as_posix(),
                    mask_relpath=mask_path.relative_to(mask_root).as_posix() if mask_path else "",
                    label=label,
                    manipulation_type=infer_manipulation_type(image_path.name) if label else "authentic",
                    source_ids=";".join(source_ids),
                    image_sha256=sha256_file(image_path) if args.hashes else "",
                    mask_sha256=sha256_file(mask_path) if args.hashes and mask_path else "",
                    width=width,
                    height=height,
                    source_revision=args.source_revision,
                )
            )
    rows = assign_connected_group_ids(rows)
    if args.dataset == "casia_v2":
        rows = deterministic_validation_split(
            rows, validation_fraction=args.validation_fraction, seed=args.seed
        )
    issues = validate_manifest(rows)
    errors = [issue for issue in issues if issue.level == "error"]
    if errors:
        raise RuntimeError("Manifest validation failed: " + "; ".join(x.message for x in errors[:10]))
    write_manifest(rows, args.output)
    authentic_count = sum(row.label == 0 for row in rows)
    tampered_count = sum(row.label == 1 for row in rows)
    paper_counts = {"casia_v2": (7491, 5063), "casia_v1": (800, 920)}
    expected = paper_counts[args.dataset]
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Counts authentic={authentic_count}, tampered={tampered_count}; paper Table 1={expected}")
    if (authentic_count, tampered_count) != expected:
        print("WARNING: dataset counts differ from EdgeCAM paper Table 1; record the dataset revision")


if __name__ == "__main__":
    main()
