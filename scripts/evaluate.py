from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from edge_finer_research.data.manifest import read_manifest
from edge_finer_research.evaluation.metrics import combined_f1, image_metrics, pixel_metrics


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--image-predictions", type=Path)
    parser.add_argument("--skip-image-metrics", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--prediction-suffix", default="__pre_sop.npy")
    parser.add_argument("--pixel-scope", choices=("fake", "all"), default="fake")
    args = parser.parse_args()
    rows = [row for row in read_manifest(args.manifest) if row.split == args.split]
    by_id = {row.sample_id: row for row in rows}
    if not rows:
        raise RuntimeError(f"No manifest rows for split={args.split}")
    measured_image = None
    if not args.skip_image_metrics:
        if args.image_predictions is None:
            raise ValueError("--image-predictions is required unless --skip-image-metrics is used")
        with args.image_predictions.open("r", newline="", encoding="utf-8") as handle:
            image_rows = list(csv.DictReader(handle))
        if {row["sample_id"] for row in image_rows} != set(by_id):
            raise RuntimeError(
                "Image prediction IDs do not exactly match the selected manifest split"
            )
        measured_image = image_metrics(
            [int(row["label"]) for row in image_rows],
            [int(row["predicted_label"]) for row in image_rows],
            [float(row["fake_score"]) for row in image_rows],
        )
    per_image: list[dict[str, object]] = []
    for row in rows:
        if args.pixel_scope == "fake" and row.label == 0:
            continue
        if row.mask_relpath:
            with Image.open(args.mask_root / row.mask_relpath) as source:
                ground_truth = np.asarray(source.convert("L")) > 0
        else:
            ground_truth = np.zeros((row.height, row.width), dtype=bool)
        prediction_path = args.prediction_dir / f"{row.sample_id}{args.prediction_suffix}"
        prediction = np.load(prediction_path, allow_pickle=False).astype(bool)
        result = pixel_metrics(ground_truth, prediction)
        per_image.append(
            {
                "sample_id": row.sample_id,
                "pixel_f1": result.f1,
                "iou": result.iou,
                "tp": result.tp,
                "tn": result.tn,
                "fp": result.fp,
                "fn": result.fn,
            }
        )
    pixel_f1 = float(np.mean([row["pixel_f1"] for row in per_image]))
    iou = float(np.mean([row["iou"] for row in per_image]))
    report = {
        "metric_protocol": {
            "image_f1": (
                "not_computed"
                if measured_image is None
                else "harmonic_mean(sensitivity,specificity)"
            ),
            "pixel_aggregation": "macro_per_image",
            "pixel_scope": args.pixel_scope,
            "mask_foreground": "nonzero",
        },
        "image_f1": measured_image.paper_f1 if measured_image is not None else None,
        "image_auc": finite_or_none(measured_image.auc) if measured_image is not None else None,
        "positive_class_f1_diagnostic": (
            measured_image.positive_class_f1 if measured_image is not None else None
        ),
        "pixel_f1": pixel_f1,
        "iou": iou,
        "c_f1": combined_f1(measured_image.paper_f1, pixel_f1) if measured_image else None,
        "image_metrics_note": (
            "Skipped explicitly; a fake-only dataset cannot measure specificity, I-F1, or AUC."
            if measured_image is None
            else None
        ),
        "samples": len(per_image),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_image[0].keys())
        writer.writeheader()
        writer.writerows(per_image)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
