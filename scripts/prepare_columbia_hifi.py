from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from PIL import Image

from edge_finer_research.data.columbia import (
    read_unique_vallist,
    threshold_mask_image,
    verify_existing_binary_png,
    write_binary_png_atomic,
)
from edge_finer_research.data.manifest import (
    ManifestRow,
    sha256_file,
    validate_manifest,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the 180 fake-only Columbia HiFi split and binary PNG masks"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--vallist", type=Path, required=True)
    parser.add_argument("--output-mask-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--image-dir", default="4cam_splc")
    parser.add_argument("--source-mask-dir", default="mask")
    parser.add_argument("--expected-count", type=int, default=180)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()

    if args.expected_count != 180:
        raise ValueError("This protocol requires exactly 180 Columbia fake samples")
    if args.output_manifest.exists():
        raise FileExistsError(args.output_manifest)
    if args.metadata_output.exists():
        raise FileExistsError(args.metadata_output)

    dataset_root = args.dataset_root.resolve()
    image_root = dataset_root / args.image_dir
    source_mask_root = dataset_root / args.source_mask_dir
    output_mask_root = args.output_mask_root.resolve()
    filenames = read_unique_vallist(args.vallist, expected_count=args.expected_count)
    rows: list[ManifestRow] = []
    created_masks = 0
    reused_masks = 0

    for filename in filenames:
        image_path = image_root / filename
        source_mask_path = source_mask_root / f"{Path(filename).stem}.jpg"
        output_mask_path = output_mask_root / f"{Path(filename).stem}.png"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if not source_mask_path.is_file():
            raise FileNotFoundError(source_mask_path)

        with Image.open(image_path) as image:
            width, height = image.size
        binary_mask, unique_values = threshold_mask_image(source_mask_path)
        if binary_mask.shape != (height, width):
            raise ValueError(
                f"Image/mask size mismatch for {filename}: "
                f"{(height, width)} vs {binary_mask.shape}"
            )
        if unique_values != {0, 255}:
            raise ValueError(
                f"Mask must contain both background and fake region: {source_mask_path}"
            )
        if output_mask_path.exists():
            verify_existing_binary_png(output_mask_path, binary_mask)
            reused_masks += 1
        else:
            write_binary_png_atomic(output_mask_path, binary_mask)
            created_masks += 1

        stem = Path(filename).stem
        rows.append(
            ManifestRow(
                sample_id=f"columbia_hifi_fake__{stem}",
                dataset="columbia_hifi_fake",
                split="test",
                group_id=f"columbia_hifi_fake__{stem}",
                image_relpath=(Path(args.image_dir) / filename).as_posix(),
                mask_relpath=output_mask_path.relative_to(output_mask_root).as_posix(),
                label=1,
                manipulation_type="splicing",
                source_ids="",
                image_sha256=sha256_file(image_path),
                mask_sha256=sha256_file(output_mask_path),
                width=width,
                height=height,
                source_revision=args.source_revision,
            )
        )

    issues = validate_manifest(
        rows,
        image_root=dataset_root,
        mask_root=output_mask_root,
        verify_hashes=True,
    )
    errors = [issue for issue in issues if issue.level == "error"]
    if errors:
        raise RuntimeError(f"Prepared Columbia manifest is invalid: {errors[:5]}")

    temporary_manifest = args.output_manifest.with_name(
        f".{args.output_manifest.name}.{uuid.uuid4().hex}.tmp"
    )
    write_manifest(rows, temporary_manifest)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary_manifest, args.output_manifest)

    report = {
        "dataset": "columbia_hifi_fake",
        "rows": len(rows),
        "split_counts": {"test": len(rows)},
        "label_counts": {"1": len(rows)},
        "vallist": str(args.vallist.resolve()),
        "vallist_sha256": sha256_file(args.vallist),
        "dataset_root": str(dataset_root),
        "image_directory": str(image_root),
        "source_mask_directory": str(source_mask_root),
        "derived_mask_directory": str(output_mask_root),
        "mask_conversion": "PIL L grayscale; foreground if value >= 128; output values 0/255",
        "created_masks": created_masks,
        "reused_verified_masks": reused_masks,
        "thumbs_db_included": False,
        "authentic_images_available": False,
        "source_revision": args.source_revision,
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
