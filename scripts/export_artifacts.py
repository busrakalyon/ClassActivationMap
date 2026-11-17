from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from edge_finer_research.data.manifest import read_manifest
from edge_finer_research.evaluation.artifacts import write_sample_artifacts
from edge_finer_research.evaluation.metrics import pixel_metrics


def resize_float(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(array.astype(np.float32), mode="F")
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--raw-cam-dir", type=Path, required=True)
    parser.add_argument("--fused-dir", type=Path, required=True)
    parser.add_argument("--post-sop-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    for row in (item for item in read_manifest(args.manifest) if item.split == args.split):
        with Image.open(args.image_root / row.image_relpath) as source:
            original = np.asarray(source.convert("RGB"))
        if row.mask_relpath:
            with Image.open(args.mask_root / row.mask_relpath) as source:
                ground_truth = np.asarray(source.convert("L")) > 0
        else:
            ground_truth = np.zeros(original.shape[:2], dtype=bool)
        with np.load(args.raw_cam_dir / f"{row.sample_id}.npz", allow_pickle=False) as archive:
            raw = resize_float(archive["cam"], (row.width, row.height))
        fused = np.load(args.fused_dir / f"{row.sample_id}.npy", allow_pickle=False)
        pre_sop = np.load(args.fused_dir / f"{row.sample_id}__pre_sop.npy", allow_pickle=False)
        post_path = (
            args.post_sop_dir / f"{row.sample_id}__post_sop.npy" if args.post_sop_dir else None
        )
        post_sop = np.load(post_path, allow_pickle=False) if post_path and post_path.is_file() else None
        before = pixel_metrics(ground_truth, pre_sop)
        metrics = {"pre_sop_pixel_f1": before.f1, "pre_sop_iou": before.iou}
        if post_sop is not None:
            after = pixel_metrics(ground_truth, post_sop)
            metrics.update({"post_sop_pixel_f1": after.f1, "post_sop_iou": after.iou})
        write_sample_artifacts(
            args.output_dir,
            row.sample_id,
            original=original,
            ground_truth=ground_truth,
            raw_cam=raw,
            epoch_fused_cam=fused,
            pre_sop_mask=pre_sop,
            post_sop_mask=post_sop,
            metrics=metrics,
        )


if __name__ == "__main__":
    main()
