from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from edge_finer_research.data.manifest import read_manifest
from edge_finer_research.evaluation.thresholds import select_pixel_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a CAM threshold on validation data only")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--fused-cam-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--candidates",
        type=float,
        nargs="+",
        default=[value / 20.0 for value in range(1, 20)],
    )
    args = parser.parse_args()
    if args.split == "test":
        raise ValueError("Threshold selection on the test split is forbidden")
    rows = [row for row in read_manifest(args.manifest) if row.split == args.split and row.label == 1]
    heatmaps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for row in rows:
        heatmaps.append(np.load(args.fused_cam_dir / f"{row.sample_id}.npy", allow_pickle=False))
        with Image.open(args.mask_root / row.mask_relpath) as source:
            masks.append(np.asarray(source.convert("L")) > 0)
    result = select_pixel_threshold(heatmaps, masks, args.candidates)
    report = {
        "selected_on_split": args.split,
        "selection_metric": "macro_pixel_f1_fake_images",
        "threshold": result.threshold,
        "pixel_f1": result.pixel_f1,
        "iou": result.iou,
        "candidates": [
            {"threshold": threshold, "pixel_f1": f1, "iou": iou}
            for threshold, f1, iou in result.scores
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
