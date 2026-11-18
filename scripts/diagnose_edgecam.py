from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from edge_finer_research.cams.epoch_fusion import fuse_epoch_cams
from edge_finer_research.config import load_toml
from edge_finer_research.data.dataset import ManifestClassificationDataset, build_transforms
from edge_finer_research.data.manifest import ManifestRow, read_manifest, validate_manifest
from edge_finer_research.diagnostics.edge_analysis import edge_comparison, pearson_correlation, sobel_edge_map
from edge_finer_research.diagnostics.edgecam_variants import (
    VariantSpec,
    compute_edgecam_diagnostic_tensor,
    default_variant_specs,
    extract_activation_gradients,
)
from edge_finer_research.diagnostics.gradient_statistics import (
    alpha_statistics,
    cam_statistics,
    distribution_statistics,
    gradient_statistics,
)
from edge_finer_research.diagnostics.localization_analysis import (
    boundary_band,
    localization_statistics,
    safe_threshold_mask,
    threshold_sweep,
)
from edge_finer_research.evaluation.metrics import pixel_metrics
from edge_finer_research.models.checkpoints import checkpoint_sha256, load_checkpoint
from edge_finer_research.models.training import build_model, seed_everything


DEFAULT_THRESHOLDS = (0.001, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5)
NORMALIZATION_EPSILON = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only numerical diagnostics for EdgeCAM Experiment 1")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results/pilot/diagnostics"))
    parser.add_argument("--run-name", help="Unique output directory name; UTC timestamp is used by default")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Deterministic diagnostic subset; omit for the full split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-consistency-count", type=int, default=8)
    parser.add_argument("--batch-atol", type=float, default=1e-5)
    parser.add_argument("--batch-rtol", type=float, default=1e-4)
    parser.add_argument("--boundary-widths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--visualization-count", type=int, default=8)
    parser.add_argument("--visualization-threshold", type=float, default=0.05)
    parser.add_argument("--keep-epoch-maps", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)


def finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(finite_json(value), indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(finite_json(rows))


def choose_rows(rows: list[ManifestRow], *, limit: int | None, seed: int) -> list[ManifestRow]:
    if limit is None or limit >= len(rows):
        return rows
    if limit < 1:
        raise ValueError("--limit must be positive")
    selected = rows.copy()
    random.Random(seed).shuffle(selected)
    # Keep both classes in ordinary pilot subsets whenever possible.
    if limit >= 2 and {row.label for row in rows} == {0, 1}:
        first_real = next(row for row in selected if row.label == 0)
        first_fake = next(row for row in selected if row.label == 1)
        remainder = [row for row in selected if row not in (first_real, first_fake)]
        selected = [first_real, first_fake, *remainder]
    return selected[:limit]


def checkpoint_paths(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("epoch_*.pt"))
    if not paths:
        raise RuntimeError(f"No epoch_*.pt checkpoints found in {directory}")
    return paths


def load_model(config: dict[str, Any], checkpoint_path: Path, device: torch.device):
    model = build_model(config, pretrained=False).to(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def load_native_image(row: ManifestRow, image_root: Path) -> np.ndarray:
    with Image.open(image_root / row.image_relpath) as source:
        return np.asarray(source.convert("RGB"))


def load_native_mask(row: ManifestRow, mask_root: Path) -> np.ndarray:
    if not row.mask_relpath:
        return np.zeros((row.height, row.width), dtype=bool)
    with Image.open(mask_root / row.mask_relpath) as source:
        mask = np.asarray(source.convert("L")) > 0
    expected = (row.height, row.width)
    if expected != (0, 0) and mask.shape != expected:
        raise RuntimeError(f"Mask shape {mask.shape} differs from manifest {expected}: {row.sample_id}")
    return mask


def resize_cam(cam: np.ndarray, height: int, width: int) -> np.ndarray:
    values = np.asarray(cam, dtype=np.float32)
    if values.shape == (height, width):
        return values
    return np.asarray(
        Image.fromarray(values, mode="F").resize((width, height), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )


def save_unit_map(path: Path, values: np.ndarray) -> None:
    clipped = np.clip(np.nan_to_num(values), 0.0, 1.0)
    Image.fromarray(np.round(clipped * 255).astype(np.uint8), mode="L").save(path)


def batch_consistency(
    model: Any,
    target_layer: Any,
    dataset: ManifestClassificationDataset,
    rows: list[ManifestRow],
    variants: list[VariantSpec],
    target_class: int,
    device: torch.device,
    count: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    count = min(count, len(rows))
    tensors = [dataset[index][0] for index in range(count)]

    def run(batch_size: int) -> dict[str, list[np.ndarray]]:
        outputs: dict[str, list[np.ndarray]] = defaultdict(list)
        for start in range(0, count, batch_size):
            images = torch.stack(tensors[start : start + batch_size]).to(device=device, dtype=torch.float32)
            logits, activation, gradients = extract_activation_gradients(model, target_layer, images, target_class)
            outputs["logits"].extend(logits.cpu().numpy())
            outputs["activation"].extend(activation.cpu().numpy())
            outputs["gradient"].extend(gradients.cpu().numpy())
            for spec in variants:
                result = compute_edgecam_diagnostic_tensor(
                    activation,
                    gradients,
                    formula=spec.formula,
                    epsilon=spec.epsilon,
                    normalization_epsilon=NORMALIZATION_EPSILON,
                )
                outputs[spec.name].extend(result.cam_normalized.cpu().numpy())
        return outputs

    one, four = run(1), run(4)
    comparisons: dict[str, Any] = {}
    overall = True
    for key in one:
        left, right = np.stack(one[key]), np.stack(four[key])
        difference = np.abs(left - right)
        passed = bool(np.allclose(left, right, atol=atol, rtol=rtol, equal_nan=False))
        overall &= passed
        comparisons[key] = {
            "allclose": passed,
            "max_abs_difference": float(difference.max(initial=0.0)),
            "mean_abs_difference": float(difference.mean()),
        }
    return {
        "passed": overall,
        "samples": count,
        "batch_sizes": [1, 4],
        "atol": atol,
        "rtol": rtol,
        "model_mode": "eval",
        "amp": False,
        "target_score_reduction": "sum_of_per_sample_fake_logits",
        "comparisons": comparisons,
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_fields = ("stage", "variant", "checkpoint", "gate", "boundary_width")
    metric_fields = (
        "pixel_f1_at_0.5",
        "iou_at_0.5",
        "inside_outside_ratio",
        "boundary_nonboundary_ratio",
        "energy_in_ground_truth_ratio",
        "cam_sobel_pearson",
        "boundary_ratio_advantage_over_sobel",
        "predicted_area_ratio_at_0.5",
        "valid_epochs",
        "raw_fused_mean_abs_difference",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key, items in groups.items():
        result = dict(zip(group_fields, key, strict=True))
        result["samples"] = len(items)
        for metric in metric_fields:
            finite = [float(item[metric]) for item in items if metric in item and np.isfinite(float(item[metric]))]
            if finite:
                result[f"mean_{metric}"] = float(np.mean(finite))
        output.append(result)
    return output


def main() -> None:
    args = parse_args()
    if args.split == "test":
        raise ValueError("Diagnostics and parameter selection on the test split are forbidden")
    if any(value <= 0 for value in args.thresholds):
        raise ValueError("All thresholds must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    config = load_toml(args.config)
    all_rows = read_manifest(args.manifest)
    issues = validate_manifest(all_rows, image_root=args.image_root, mask_root=args.mask_root)
    errors = [issue for issue in issues if issue.level == "error"]
    if errors:
        raise RuntimeError(f"Manifest validation failed: {errors[:10]}")
    rows = [row for row in all_rows if row.split == args.split]
    if not rows:
        raise RuntimeError(f"No manifest rows for split={args.split}")
    rows = choose_rows(rows, limit=args.limit, seed=args.seed)
    fake_rows = [row for row in rows if row.label == 1]
    if not fake_rows:
        raise RuntimeError("Diagnostics require at least one fake sample with ground truth")
    checkpoints = checkpoint_paths(args.checkpoint_dir)
    variants = default_variant_specs()
    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_root / safe_name(run_name)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic run: {output}")
    output.mkdir(parents=True)
    cache_root = output / "epoch_maps"
    cache_root.mkdir()
    visualization_root = output / "visualizations"
    visualization_root.mkdir()
    shutil.copy2(args.config, output / "resolved_config.toml")
    seed_everything(args.seed)

    transform = build_transforms(config, train=False)
    dataset = ManifestClassificationDataset(rows, args.image_root, transform)
    target_class = int(config["experiment"]["positive_class"])
    image_size = int(config["experiment"]["image_size"])
    gate_threshold = float(config["edgecam"]["classification_gate"])
    gradient_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    cam_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    checkpoint_metadata: list[dict[str, Any]] = []
    native_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def native_inputs(row: ManifestRow) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if row.sample_id not in native_cache:
            original = load_native_image(row, args.image_root)
            native_cache[row.sample_id] = (
                original,
                load_native_mask(row, args.mask_root),
                sobel_edge_map(original).astype(np.float32),
            )
        return native_cache[row.sample_id]

    for checkpoint_path in checkpoints:
        checkpoint_name = checkpoint_path.stem
        checkpoint_metadata.append({"name": checkpoint_name, "sha256": checkpoint_sha256(checkpoint_path)})
        model = load_model(config, checkpoint_path, device)
        target_layer = model.resolve_target_layer(str(config["model"]["target_layer"]))
        for start in range(0, len(rows), args.batch_size):
            indices = range(start, min(start + args.batch_size, len(rows)))
            images = torch.stack([dataset[index][0] for index in indices]).to(device=device, dtype=torch.float32)
            logits, activation, gradients = extract_activation_gradients(model, target_layer, images, target_class)
            probabilities = logits.softmax(dim=1)
            for local_index, row_index in enumerate(indices):
                row = rows[row_index]
                grad = gradients[local_index].cpu().numpy()
                activation_values = activation[local_index].cpu().numpy()
                gradient_rows.append(
                    {
                        "checkpoint": checkpoint_name,
                        "sample_id": row.sample_id,
                        "label": row.label,
                        "fake_score": float(probabilities[local_index, target_class]),
                        "predicted_class": int(logits[local_index].argmax()),
                        **gradient_statistics(grad),
                        **distribution_statistics(activation_values, prefix="activation_"),
                        **distribution_statistics(
                            activation_values.sum(axis=(-2, -1)), prefix="activation_spatial_sum_"
                        ),
                    }
                )
            for spec in variants:
                result = compute_edgecam_diagnostic_tensor(
                    activation,
                    gradients,
                    formula=spec.formula,
                    epsilon=spec.epsilon,
                    normalization_epsilon=NORMALIZATION_EPSILON,
                )
                resized = F.interpolate(
                    result.cam_normalized[:, None],
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
                for local_index, row_index in enumerate(indices):
                    row = rows[row_index]
                    valid = float(probabilities[local_index, target_class]) >= gate_threshold
                    variant_dir = cache_root / checkpoint_name / spec.name
                    variant_dir.mkdir(parents=True, exist_ok=True)
                    map512 = resized[local_index].cpu().numpy().astype(np.float32)
                    np.savez_compressed(
                        variant_dir / f"{safe_name(row.sample_id)}.npz",
                        cam=map512,
                        valid_for_epoch_fusion=valid,
                        fake_score=float(probabilities[local_index, target_class]),
                    )
                    view = SimpleNamespace(
                        cam_pre_relu=result.cam_pre_relu[local_index].cpu().numpy(),
                        cam_post_relu=result.cam_post_relu[local_index].cpu().numpy(),
                        cam_normalized=result.cam_normalized[local_index].cpu().numpy(),
                        channel_contributions=result.channel_contributions[local_index].cpu().numpy(),
                    )
                    alpha_rows.append(
                        {
                            "checkpoint": checkpoint_name,
                            "sample_id": row.sample_id,
                            "variant": spec.name,
                            "formula": spec.formula,
                            "denominator_epsilon": spec.epsilon,
                            "normalization_epsilon": NORMALIZATION_EPSILON,
                            **alpha_statistics(
                                result.alpha[local_index].cpu().numpy(),
                                result.raw_alpha[local_index].cpu().numpy(),
                                result.suppressed_by_epsilon[local_index].cpu().numpy(),
                                gradients[local_index].cpu().numpy(),
                                result.denominator[local_index].cpu().numpy(),
                            ),
                        }
                    )
                    cam_rows.append(
                        {
                            "checkpoint": checkpoint_name,
                            "sample_id": row.sample_id,
                            "variant": spec.name,
                            "formula": spec.formula,
                            "denominator_epsilon": spec.epsilon,
                            **cam_statistics(view, active_thresholds=(0.001, 0.01, 0.05, 0.5)),
                        }
                    )
                    if row.label == 1:
                        native_cam = resize_cam(map512, row.height, row.width)
                        _original, ground_truth, sobel = native_inputs(row)
                        measured = pixel_metrics(ground_truth, safe_threshold_mask(native_cam, 0.5))
                        for width in args.boundary_widths:
                            per_image_rows.append(
                                {
                                    "stage": "epoch",
                                    "checkpoint": checkpoint_name,
                                    "variant": spec.name,
                                    "gate": "not_applied_single_epoch",
                                    "sample_id": row.sample_id,
                                    "boundary_width": width,
                                    "valid_for_gate": valid,
                                    "pixel_f1_at_0.5": measured.f1,
                                    "iou_at_0.5": measured.iou,
                                    "predicted_area_ratio_at_0.5": float(safe_threshold_mask(native_cam, 0.5).mean()),
                                    **localization_statistics(native_cam, ground_truth, boundary_width=width),
                                    **edge_comparison(native_cam, sobel, ground_truth, boundary_width=width),
                                }
                            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Batch check uses the final epoch checkpoint and exactly the same evaluation transform.
    consistency_model = load_model(config, checkpoints[-1], device)
    consistency_layer = consistency_model.resolve_target_layer(str(config["model"]["target_layer"]))
    batch_report = batch_consistency(
        consistency_model,
        consistency_layer,
        dataset,
        rows,
        variants,
        target_class,
        device,
        args.batch_consistency_count,
        args.batch_atol,
        args.batch_rtol,
    )
    write_json(output / "batch_consistency.json", batch_report)
    del consistency_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    threshold_inputs: dict[tuple[str, str], tuple[list[np.ndarray], list[np.ndarray]]] = defaultdict(lambda: ([], []))
    visual_ids = {row.sample_id for row in fake_rows[: args.visualization_count]}
    for row in fake_rows:
        original, ground_truth, sobel = native_inputs(row)
        for spec in variants:
            maps: list[np.ndarray] = []
            flags: list[bool] = []
            for checkpoint_path in checkpoints:
                path = cache_root / checkpoint_path.stem / spec.name / f"{safe_name(row.sample_id)}.npz"
                with np.load(path, allow_pickle=False) as archive:
                    maps.append(archive["cam"])
                    flags.append(bool(archive["valid_for_epoch_fusion"]))
            raw_last = resize_cam(maps[-1], row.height, row.width)
            for gate_name, gate_flags in (("on", flags), ("off", [True] * len(flags))):
                fused_result = fuse_epoch_cams(maps, valid=gate_flags, method="max")
                fused = resize_cam(fused_result.map, row.height, row.width)
                prediction = safe_threshold_mask(fused, 0.5)
                measured = pixel_metrics(ground_truth, prediction)
                threshold_inputs[(spec.name, gate_name)][0].append(fused)
                threshold_inputs[(spec.name, gate_name)][1].append(ground_truth)
                for width in args.boundary_widths:
                    per_image_rows.append(
                        {
                            "stage": "fused",
                            "checkpoint": "all_epochs_max",
                            "variant": spec.name,
                            "gate": gate_name,
                            "sample_id": row.sample_id,
                            "boundary_width": width,
                            "valid_epochs": fused_result.valid_count,
                            "total_epochs": fused_result.total_count,
                            "pixel_f1_at_0.5": measured.f1,
                            "iou_at_0.5": measured.iou,
                            "predicted_area_ratio_at_0.5": float(prediction.mean()),
                            "raw_fused_mean_abs_difference": float(np.mean(np.abs(raw_last - fused))),
                            "raw_fused_pearson": pearson_correlation(raw_last, fused),
                            **localization_statistics(fused, ground_truth, boundary_width=width),
                            **edge_comparison(fused, sobel, ground_truth, boundary_width=width),
                        }
                    )
                if row.sample_id in visual_ids:
                    sample_dir = visualization_root / safe_name(row.sample_id)
                    sample_dir.mkdir(exist_ok=True)
                    Image.fromarray(original).save(sample_dir / "original.png")
                    save_unit_map(sample_dir / "ground_truth.png", ground_truth)
                    save_unit_map(sample_dir / f"ground_truth_boundary_w{args.boundary_widths[0]}.png", boundary_band(ground_truth, args.boundary_widths[0]))
                    save_unit_map(sample_dir / "sobel.png", sobel)
                    save_unit_map(sample_dir / f"{spec.name}__gate_{gate_name}__fused.png", fused)
                    save_unit_map(
                        sample_dir / f"{spec.name}__gate_{gate_name}__threshold_{args.visualization_threshold:g}.png",
                        safe_threshold_mask(fused, args.visualization_threshold),
                    )

    threshold_rows: list[dict[str, Any]] = []
    for (variant, gate), (cams, masks) in threshold_inputs.items():
        for item in threshold_sweep(cams, masks, args.thresholds):
            threshold_rows.append({"variant": variant, "gate": gate, **item})

    variant_rows = aggregate_rows(per_image_rows)
    fused_gate_rows = [row for row in per_image_rows if row["stage"] == "fused" and row["boundary_width"] == args.boundary_widths[0]]
    gate_report: dict[str, Any] = {}
    for spec in variants:
        gate_report[spec.name] = {}
        for gate in ("on", "off"):
            items = [row for row in fused_gate_rows if row["variant"] == spec.name and row["gate"] == gate]
            gate_report[spec.name][gate] = {
                "samples": len(items),
                "mean_valid_epochs": float(np.mean([item["valid_epochs"] for item in items])),
                "mean_pixel_f1_at_0.5": float(np.mean([item["pixel_f1_at_0.5"] for item in items])),
                "mean_iou_at_0.5": float(np.mean([item["iou_at_0.5"] for item in items])),
                "mean_boundary_nonboundary_ratio": float(np.mean([item["boundary_nonboundary_ratio"] for item in items])),
            }
    write_json(output / "gate_comparison.json", gate_report)
    write_csv(output / "gradient_statistics.csv", gradient_rows)
    write_csv(output / "alpha_statistics.csv", alpha_rows)
    write_csv(output / "cam_statistics.csv", cam_rows)
    write_csv(output / "per_image_metrics.csv", per_image_rows)
    write_csv(output / "variant_comparison.csv", variant_rows)
    write_csv(output / "threshold_analysis.csv", threshold_rows)

    best_thresholds: dict[str, Any] = {}
    for spec in variants:
        for gate in ("on", "off"):
            candidates = [
                row for row in threshold_rows
                if row["variant"] == spec.name and row["gate"] == gate and row["condition"] == "threshold"
            ]
            best = max(candidates, key=lambda item: (item["pixel_f1"], item["iou"], item["threshold"]))
            best_thresholds[f"{spec.name}__gate_{gate}"] = best
    numerical_summary: dict[str, Any] = {}
    for spec in variants:
        alpha_items = [row for row in alpha_rows if row["variant"] == spec.name]
        cam_items = [row for row in cam_rows if row["variant"] == spec.name]
        numerical_summary[spec.name] = {
            "mean_epsilon_suppressed_ratio": float(
                np.mean([item["epsilon_suppressed_ratio"] for item in alpha_items])
            ),
            "mean_positive_gradient_epsilon_suppressed_ratio": float(
                np.mean([item["positive_gradient_epsilon_suppressed_ratio"] for item in alpha_items])
            ),
            "total_raw_alpha_nan_count": int(sum(item["raw_alpha_nan_count"] for item in alpha_items)),
            "total_raw_alpha_inf_count": int(
                sum(item["raw_alpha_posinf_count"] + item["raw_alpha_neginf_count"] for item in alpha_items)
            ),
            "mean_normalized_zero_pixel_ratio": float(
                np.mean([item["normalized_zero_pixel_ratio"] for item in cam_items])
            ),
            "mean_negative_channel_contribution_fraction": float(
                np.mean([item["negative_channel_contribution_fraction"] for item in cam_items])
            ),
        }
    summary = {
        "status": "completed",
        "scope": "CASIAv2 validation diagnostics only",
        "scientific_result": False,
        "split": args.split,
        "samples": len(rows),
        "fake_samples": len(fake_rows),
        "subset_limit": args.limit,
        "seed": args.seed,
        "device": str(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "checkpoints": checkpoint_metadata,
        "variants": [spec.__dict__ for spec in variants],
        "normalization_epsilon_fixed": NORMALIZATION_EPSILON,
        "computation_dtype": "float32",
        "amp_used_for_diagnostics": False,
        "boundary_definition": "3x3 dilation minus 3x3 erosion; width equals iteration count",
        "boundary_widths": args.boundary_widths,
        "thresholds": args.thresholds,
        "fake_class_gate_threshold": gate_threshold,
        "batch_consistency": batch_report,
        "numerical_summary": numerical_summary,
        "best_validation_thresholds_diagnostic_only": best_thresholds,
        "ground_truth_usage": "evaluation_only_not_forward_not_CAM_not_loss",
        "processing_order": [
            "gradient", "alpha", "spatial_weight", "channel_sum", "final_relu",
            "per_image_minmax", "resize_to_model_input", "epoch_max_fusion",
            "resize_to_native_coordinates", "positive_pixel_threshold",
        ],
    }
    write_json(output / "summary.json", summary)
    if not args.keep_epoch_maps:
        shutil.rmtree(cache_root)
    print(json.dumps({"status": "completed", "output": str(output), "samples": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
