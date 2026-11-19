from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from edge_finer_research.data.manifest import read_manifest
from edge_finer_research.evaluation.metrics import pixel_metrics
from edge_finer_research.evaluation.rethreshold import (
    measure_thresholds,
    reference_masks,
    threshold_mask,
)


DEFAULT_THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-threshold existing float32 fused CAMs without CAM recomputation"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--fused-cam-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split != "val":
        raise ValueError("Threshold selection is restricted to split=val")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output_dir / "masks"
    prediction_dir.mkdir()

    rows = [row for row in read_manifest(args.manifest) if row.split == args.split]
    fake_rows = [row for row in rows if row.label == 1]
    if not rows or not fake_rows:
        raise RuntimeError("Selected validation split must contain samples and fake samples")

    cams_by_id: dict[str, np.ndarray] = {}
    for row in rows:
        path = args.fused_cam_dir / f"{row.sample_id}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        cam = np.load(path, allow_pickle=False)
        if cam.dtype != np.float32:
            raise TypeError(f"Expected float32 fused CAM, found {cam.dtype}: {path}")
        if cam.shape != (row.height, row.width):
            raise ValueError(f"CAM shape mismatch for {row.sample_id}: {cam.shape}")
        if not np.isfinite(cam).all():
            raise ValueError(f"CAM contains NaN/Inf: {path}")
        cams_by_id[row.sample_id] = cam

    truths: list[np.ndarray] = []
    fake_cams: list[np.ndarray] = []
    for row in fake_rows:
        if not row.mask_relpath:
            raise RuntimeError(f"Fake sample has no ground-truth mask: {row.sample_id}")
        with Image.open(args.mask_root / row.mask_relpath) as source:
            truth = np.asarray(source.convert("L")) > 0
        cam = cams_by_id[row.sample_id]
        if truth.shape != cam.shape:
            raise ValueError(
                f"Ground-truth/CAM shape mismatch for {row.sample_id}: {truth.shape} vs {cam.shape}"
            )
        truths.append(truth)
        fake_cams.append(cam)

    scores = measure_thresholds(fake_cams, truths, args.thresholds)
    best = max(scores, key=lambda item: (item.pixel_f1, item.iou, -item.threshold))
    references = reference_masks(truths)

    threshold_rows = []
    for item in scores:
        all_validation_area = float(
            np.mean(
                [
                    threshold_mask(cams_by_id[row.sample_id], item.threshold).mean()
                    for row in rows
                ]
            )
        )
        threshold_rows.append(
            {
                "threshold": item.threshold,
                "pixel_f1": item.pixel_f1,
                "iou": item.iou,
                "mean_predicted_area_ratio": item.predicted_area_ratio,
                "mean_predicted_area_ratio_all_validation": all_validation_area,
                "mean_ground_truth_area_ratio": item.ground_truth_area_ratio,
            }
        )
    with (args.output_dir / "threshold_analysis.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=threshold_rows[0].keys())
        writer.writeheader()
        writer.writerows(threshold_rows)

    per_image: list[dict[str, object]] = []
    for row in rows:
        prediction = threshold_mask(cams_by_id[row.sample_id], best.threshold)
        np.save(prediction_dir / f"{row.sample_id}__pre_sop.npy", prediction, allow_pickle=False)
        if row.label == 1:
            with Image.open(args.mask_root / row.mask_relpath) as source:
                truth = np.asarray(source.convert("L")) > 0
            metrics = pixel_metrics(truth, prediction)
            per_image.append(
                {
                    "sample_id": row.sample_id,
                    "threshold": best.threshold,
                    "pixel_f1": metrics.f1,
                    "iou": metrics.iou,
                    "predicted_area_ratio": float(prediction.mean()),
                    "ground_truth_area_ratio": float(truth.mean()),
                    "tp": metrics.tp,
                    "tn": metrics.tn,
                    "fp": metrics.fp,
                    "fn": metrics.fn,
                }
            )
    with (args.output_dir / "per_image_selected.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=per_image[0].keys())
        writer.writeheader()
        writer.writerows(per_image)

    report = {
        "selected_on_split": args.split,
        "selection_scope": "fake_images_only",
        "selection_metric": "macro_pixel_f1_fake_images",
        "tie_break": "higher_iou_then_lower_threshold",
        "cam_source": str(args.fused_cam_dir.resolve()),
        "cam_dtype_required": "float32",
        "comparison_operator": ">=",
        "threshold": best.threshold,
        "pixel_f1": best.pixel_f1,
        "iou": best.iou,
        "mean_predicted_area_ratio": best.predicted_area_ratio,
        "mean_predicted_area_ratio_all_validation": next(
            row["mean_predicted_area_ratio_all_validation"]
            for row in threshold_rows
            if row["threshold"] == best.threshold
        ),
        "mean_ground_truth_area_ratio": best.ground_truth_area_ratio,
        "samples": {"all_validation": len(rows), "fake_localization": len(fake_rows)},
        "references": references,
        "candidates": [
            {
                "threshold": row["threshold"],
                "pixel_f1": row["pixel_f1"],
                "iou": row["iou"],
            }
            for row in threshold_rows
        ],
        "thresholds": threshold_rows,
        "prediction_dir": str(prediction_dir.resolve()),
        "sop_applied": False,
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
