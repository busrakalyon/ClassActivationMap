from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from edge_finer_research.data.manifest import read_manifest


def unit_to_rgb(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    red = clipped
    green = 1.0 - np.abs(2.0 * clipped - 1.0)
    blue = 1.0 - clipped
    return np.rint(np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=bool).astype(np.uint8) * 255
    return np.repeat(values[:, :, None], 3, axis=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize fused CAM and MCG-SOP selections")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--fused-cam-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample-count", type=int, default=20)
    args = parser.parse_args()
    if args.sample_count < 1:
        raise ValueError("sample-count must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        row
        for row in read_manifest(args.manifest)
        if row.split == args.split and row.label == 1 and row.mask_relpath
    ][: args.sample_count]
    if not rows:
        raise RuntimeError("No fake samples with ground-truth masks for visualization")

    sample_ids: list[str] = []
    for row in rows:
        with Image.open(args.image_root / row.image_relpath) as source:
            original = np.asarray(source.convert("RGB"))
        with Image.open(args.mask_root / row.mask_relpath) as source:
            ground_truth = np.asarray(source.convert("L")) > 0
        cam = np.load(args.fused_cam_dir / f"{row.sample_id}.npy", allow_pickle=False)
        prediction = np.load(
            args.prediction_dir / f"{row.sample_id}__post_sop.npy", allow_pickle=False
        ).astype(bool)
        expected_shape = original.shape[:2]
        if cam.shape != expected_shape or ground_truth.shape != expected_shape:
            raise ValueError(f"Input shape mismatch for {row.sample_id}")
        if prediction.shape != expected_shape:
            raise ValueError(f"Prediction shape mismatch for {row.sample_id}")
        if not np.isfinite(cam).all():
            raise ValueError(f"CAM contains NaN/Inf for {row.sample_id}")

        sample_dir = args.output_dir / row.sample_id
        sample_dir.mkdir()
        panels = [original, unit_to_rgb(cam), mask_to_rgb(prediction), mask_to_rgb(ground_truth)]
        names = ["original.png", "fused_cam.png", "selected_sop.png", "ground_truth.png"]
        for name, panel in zip(names, panels, strict=True):
            Image.fromarray(panel).save(sample_dir / name)
        comparison = np.concatenate(panels, axis=1)
        Image.fromarray(comparison).save(sample_dir / "comparison.png")
        sample_ids.append(row.sample_id)

    report = {
        "split": args.split,
        "samples": sample_ids,
        "panel_order": ["original", "fused_cam", "selected_sop", "ground_truth"],
        "cam_color_mapping": "fixed_blue_green_red_0_to_1",
        "visual_transform_affects_metrics": False,
    }
    (args.output_dir / "visualization_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
