from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Required by deterministic CUDA GEMM. It must be set before the first CUDA operation.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F

from edge_finer_research.cams.edgecam import paper_edgecam_tensor
from edge_finer_research.config import load_toml
from edge_finer_research.data.dataset import ManifestClassificationDataset, build_transforms
from edge_finer_research.data.manifest import read_manifest
from edge_finer_research.diagnostics.batch_consistency import (
    CUDA_PROFILES,
    backend_state,
    capture_forward_gradient,
    cuda_backend_profile,
    mask_iou,
    model_eval_audit,
    numerical_comparison,
)
from edge_finer_research.diagnostics.edgecam_variants import compute_edgecam_diagnostic_tensor
from edge_finer_research.models.checkpoints import checkpoint_sha256, load_checkpoint
from edge_finer_research.models.training import build_model, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolate batch-shape sensitivity in EdgeCAM")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results/pilot/batch_diagnostics"))
    parser.add_argument("--run-name")
    parser.add_argument("--split", default="val")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat-count", type=int, default=3)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.5])
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(CUDA_PROFILES),
        default=list(CUDA_PROFILES),
    )
    parser.add_argument("--include-cpu-float64", action="store_true")
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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(finite_json(rows))


def load_model(config: dict[str, Any], checkpoint_path: Path, device: torch.device, dtype: torch.dtype):
    model = build_model(config, pretrained=False).to(device=device, dtype=dtype)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def observed_layers(model: Any, target_layer: Any) -> dict[str, Any]:
    layers: dict[str, Any] = {"target": target_layer}
    for index, module in enumerate(model.backbone.features):
        layers[f"backbone.features.{index}"] = module
    layers.update(
        {
            "backbone.avgpool": model.backbone.avgpool,
            "backbone.classifier": model.backbone.classifier,
            "embedding_linear": model.embedding,
            "embedding_output": model.dropout_after_embedding,
        }
    )
    return layers


def run_batches(
    model: Any,
    target_layer: Any,
    layers: dict[str, Any],
    tensors: list[torch.Tensor],
    *,
    batch_size: int,
    target_class: int,
    image_size: int,
    epsilon: float,
    gate_threshold: float,
    device: torch.device,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> tuple[dict[str, list[np.ndarray]], list[dict[str, Any]]]:
    outputs: dict[str, list[np.ndarray]] = defaultdict(list)
    metadata: list[dict[str, Any]] = []
    for start in range(0, len(tensors), batch_size):
        images = torch.stack(tensors[start : start + batch_size]).to(device=device, dtype=dtype)
        captured = capture_forward_gradient(model, target_layer, layers, images, target_class)
        activation = captured["target"]
        gradient = captured["gradient"]
        main_cam = paper_edgecam_tensor(
            activation,
            gradient,
            epsilon=epsilon,
            apply_relu=True,
            normalize=True,
        )
        diagnostic = compute_edgecam_diagnostic_tensor(
            activation,
            gradient,
            formula="original",
            epsilon=epsilon,
            normalization_epsilon=epsilon,
        )
        resized = F.interpolate(
            main_cam[:, None], (image_size, image_size), mode="bilinear", align_corners=False
        )[:, 0]
        probabilities = captured["logits"].softmax(dim=1)
        for key, values in captured.items():
            outputs[key].extend(values.cpu().numpy())
        outputs["cam_main"].extend(main_cam.cpu().numpy())
        outputs["cam_resized"].extend(resized.cpu().numpy())
        outputs["cam_pre_relu"].extend(diagnostic.cam_pre_relu.cpu().numpy())
        outputs["cam_post_relu"].extend(diagnostic.cam_post_relu.cpu().numpy())
        outputs["cam_diagnostic_original"].extend(diagnostic.cam_normalized.cpu().numpy())
        for index in range(images.shape[0]):
            equivalence = numerical_comparison(
                main_cam[index].cpu().numpy(),
                diagnostic.cam_normalized[index].cpu().numpy(),
                atol=atol,
                rtol=rtol,
            )
            metadata.append(
                {
                    "fake_score": float(probabilities[index, target_class]),
                    "gate": bool(float(probabilities[index, target_class]) >= gate_threshold),
                    "predicted_class": int(captured["logits"][index].argmax()),
                    "main_diagnostic_equivalence": equivalence,
                }
            )
    return outputs, metadata


def compare_runs(
    profile: str,
    sample_ids: list[str],
    one: dict[str, list[np.ndarray]],
    four: dict[str, list[np.ndarray]],
    one_meta: list[dict[str, Any]],
    four_meta: list[dict[str, Any]],
    *,
    thresholds: list[float],
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for component in one:
        for index, sample_id in enumerate(sample_ids):
            measured = numerical_comparison(one[component][index], four[component][index], atol=atol, rtol=rtol)
            row: dict[str, Any] = {
                "profile": profile,
                "comparison": "batch_1_vs_4",
                "sample_id": sample_id,
                "component": component,
                **measured,
            }
            if component in {"cam_main", "cam_resized", "cam_diagnostic_original"}:
                for threshold in thresholds:
                    row[f"mask_iou_at_{threshold:g}"] = mask_iou(
                        one[component][index], four[component][index], threshold
                    )
            rows.append(row)
    for index, sample_id in enumerate(sample_ids):
        rows.append(
            {
                "profile": profile,
                "comparison": "batch_1_vs_4",
                "sample_id": sample_id,
                "component": "fake_class_gate",
                "allclose": one_meta[index]["gate"] == four_meta[index]["gate"],
                "gate_batch_1": one_meta[index]["gate"],
                "gate_batch_4": four_meta[index]["gate"],
                "fake_score_batch_1": one_meta[index]["fake_score"],
                "fake_score_batch_4": four_meta[index]["fake_score"],
                "fake_score_abs_difference": abs(
                    one_meta[index]["fake_score"] - four_meta[index]["fake_score"]
                ),
            }
        )
        for batch_label, metadata in (("batch_1", one_meta[index]), ("batch_4", four_meta[index])):
            rows.append(
                {
                    "profile": profile,
                    "comparison": "main_vs_diagnostic_original",
                    "sample_id": sample_id,
                    "component": batch_label,
                    **metadata["main_diagnostic_equivalence"],
                }
            )
    return rows


def isolated_arcface_comparison(
    model: Any,
    canonical_embeddings: list[np.ndarray],
    device: torch.device,
    dtype: torch.dtype,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    embeddings = torch.from_numpy(np.stack(canonical_embeddings)).to(device=device, dtype=dtype)
    with torch.no_grad():
        individual = torch.cat([model.arcface.forward_logits(item[None]) for item in embeddings])
        batched = model.arcface.forward_logits(embeddings)
    return numerical_comparison(individual.cpu().numpy(), batched.cpu().numpy(), atol=atol, rtol=rtol)


def repeat_batch_one(
    model: Any,
    target_layer: Any,
    layers: dict[str, Any],
    tensor: torch.Tensor,
    repeats: int,
    run_kwargs: dict[str, Any],
    *,
    profile: str,
    sample_id: str,
    thresholds: list[float],
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
    runs = [run_batches(model, target_layer, layers, [tensor], batch_size=1, **run_kwargs)[0] for _ in range(repeats)]
    rows: list[dict[str, Any]] = []
    for repeat_index in range(1, repeats):
        for component in runs[0]:
            measured = numerical_comparison(
                runs[0][component][0], runs[repeat_index][component][0], atol=atol, rtol=rtol
            )
            row: dict[str, Any] = {
                "profile": profile,
                "comparison": f"batch_1_repeat_0_vs_{repeat_index}",
                "sample_id": sample_id,
                "component": component,
                **measured,
            }
            if component in {"cam_main", "cam_resized", "cam_diagnostic_original"}:
                for threshold in thresholds:
                    row[f"mask_iou_at_{threshold:g}"] = mask_iou(
                        runs[0][component][0], runs[repeat_index][component][0], threshold
                    )
            rows.append(row)
    return rows


def profile_summary(profile: str, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    profile_rows = [row for row in rows if row["profile"] == profile]
    batch_rows = [row for row in profile_rows if row["comparison"] == "batch_1_vs_4"]
    components: dict[str, Any] = {}
    for component in sorted({row["component"] for row in batch_rows}):
        items = [row for row in batch_rows if row["component"] == component]
        numeric = [row for row in items if "max_abs_difference" in row]
        components[component] = {
            "all_samples_close": all(bool(row["allclose"]) for row in items),
            "max_abs_difference": max((row["max_abs_difference"] for row in numeric), default=None),
            "mean_abs_difference": float(np.mean([row["mean_abs_difference"] for row in numeric])) if numeric else None,
            "mean_pearson": float(np.mean([row["pearson"] for row in numeric])) if numeric else None,
        }
        iou_fields = sorted({key for row in items for key in row if key.startswith("mask_iou_at_")})
        for field in iou_fields:
            values = [float(row[field]) for row in items if field in row]
            components[component][f"minimum_{field}"] = min(values)
            components[component][f"mean_{field}"] = float(np.mean(values))
    repeat_rows = [row for row in profile_rows if row["comparison"].startswith("batch_1_repeat")]
    forward_order = [
        *(f"backbone.features.{index}" for index in range(9)),
        "backbone.avgpool",
        "backbone.classifier",
        "embedding_linear",
        "embedding_output",
        "logits",
    ]
    first_forward_difference = next(
        (
            component
            for component in forward_order
            if component in components and not components[component]["all_samples_close"]
        ),
        None,
    )
    return {
        "backend": metadata,
        "components": components,
        "gate_agreement": all(bool(row["allclose"]) for row in batch_rows if row["component"] == "fake_class_gate"),
        "batch_one_repeat_consistent": all(bool(row["allclose"]) for row in repeat_rows),
        "first_forward_component_not_close": first_forward_difference,
    }


def main() -> None:
    args = parse_args()
    if args.split == "test":
        raise ValueError("Batch diagnostics must not be selected or tuned on the test split")
    if args.samples < 4 or args.repeat_count < 2:
        raise ValueError("--samples must be at least four for a real batch-1 versus batch-4 comparison")
    config = load_toml(args.config)
    rows = [row for row in read_manifest(args.manifest) if row.split == args.split]
    if not rows:
        raise RuntimeError(f"No manifest rows for split={args.split}")
    random.Random(args.seed).shuffle(rows)
    rows = rows[: min(args.samples, len(rows))]
    seed_everything(args.seed)
    dataset = ManifestClassificationDataset(rows, args.image_root, build_transforms(config, train=False))
    tensors = [dataset[index][0] for index in range(len(dataset))]
    sample_ids = [row.sample_id for row in rows]
    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_root / safe_name(run_name)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)

    target_class = int(config["experiment"]["positive_class"])
    image_size = int(config["experiment"]["image_size"])
    epsilon = float(config["edgecam"]["epsilon"])
    gate_threshold = float(config["edgecam"]["classification_gate"])
    rows_out: list[dict[str, Any]] = []
    profile_metadata: dict[str, Any] = {}
    model_audit: dict[str, Any] | None = None

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested backend matrix")
    model = load_model(config, args.checkpoint, device, torch.float32)
    target_layer = model.resolve_target_layer(str(config["model"]["target_layer"]))
    layers = observed_layers(model, target_layer)
    model_audit = model_eval_audit(model)
    initial_backend = backend_state()
    common = {
        "target_class": target_class,
        "image_size": image_size,
        "epsilon": epsilon,
        "gate_threshold": gate_threshold,
        "device": device,
        "dtype": torch.float32,
        "atol": args.atol,
        "rtol": args.rtol,
    }
    for profile in args.profiles:
        with cuda_backend_profile(profile) as active_backend:
            one, one_meta = run_batches(model, target_layer, layers, tensors, batch_size=1, **common)
            four, four_meta = run_batches(model, target_layer, layers, tensors, batch_size=4, **common)
            rows_out.extend(
                compare_runs(
                    profile,
                    sample_ids,
                    one,
                    four,
                    one_meta,
                    four_meta,
                    thresholds=args.thresholds,
                    atol=args.atol,
                    rtol=args.rtol,
                )
            )
            rows_out.extend(
                repeat_batch_one(
                    model,
                    target_layer,
                    layers,
                    tensors[0],
                    args.repeat_count,
                    common,
                    profile=profile,
                    sample_id=sample_ids[0],
                    thresholds=args.thresholds,
                    atol=args.atol,
                    rtol=args.rtol,
                )
            )
            profile_metadata[profile] = {
                "device": str(device),
                "dtype": "float32",
                "backend": active_backend,
                "isolated_arcface": isolated_arcface_comparison(
                    model,
                    one["embedding_output"],
                    device,
                    torch.float32,
                    atol=args.atol,
                    rtol=args.rtol,
                ),
                "main_diagnostic_equivalent_batch_1": all(
                    item["main_diagnostic_equivalence"]["allclose"] for item in one_meta
                ),
                "main_diagnostic_equivalent_batch_4": all(
                    item["main_diagnostic_equivalence"]["allclose"] for item in four_meta
                ),
            }
    del model
    torch.cuda.empty_cache()

    if args.include_cpu_float64:
        profile = "cpu_float64_reference"
        cpu = torch.device("cpu")
        cpu_model = load_model(config, args.checkpoint, cpu, torch.float64)
        cpu_target = cpu_model.resolve_target_layer(str(config["model"]["target_layer"]))
        cpu_layers = observed_layers(cpu_model, cpu_target)
        cpu_common = {
            **common,
            "device": cpu,
            "dtype": torch.float64,
        }
        one, one_meta = run_batches(cpu_model, cpu_target, cpu_layers, tensors, batch_size=1, **cpu_common)
        four, four_meta = run_batches(cpu_model, cpu_target, cpu_layers, tensors, batch_size=4, **cpu_common)
        rows_out.extend(
            compare_runs(
                profile,
                sample_ids,
                one,
                four,
                one_meta,
                four_meta,
                thresholds=args.thresholds,
                atol=args.atol,
                rtol=args.rtol,
            )
        )
        rows_out.extend(
            repeat_batch_one(
                cpu_model,
                cpu_target,
                cpu_layers,
                tensors[0],
                args.repeat_count,
                cpu_common,
                profile=profile,
                sample_id=sample_ids[0],
                thresholds=args.thresholds,
                atol=args.atol,
                rtol=args.rtol,
            )
        )
        profile_metadata[profile] = {
            "device": "cpu",
            "dtype": "float64",
            "isolated_arcface": isolated_arcface_comparison(
                cpu_model,
                one["embedding_output"],
                cpu,
                torch.float64,
                atol=args.atol,
                rtol=args.rtol,
            ),
            "main_diagnostic_equivalent_batch_1": all(
                item["main_diagnostic_equivalence"]["allclose"] for item in one_meta
            ),
            "main_diagnostic_equivalent_batch_4": all(
                item["main_diagnostic_equivalence"]["allclose"] for item in four_meta
            ),
        }

    summaries = {
        profile: profile_summary(profile, rows_out, metadata)
        for profile, metadata in profile_metadata.items()
    }
    report = {
        "status": "completed",
        "scientific_result": False,
        "purpose": "batch-shape numerical diagnosis only",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "samples": sample_ids,
        "split": args.split,
        "batch_sizes": [1, 4],
        "thresholds": args.thresholds,
        "atol": args.atol,
        "rtol": args.rtol,
        "epsilon_unchanged": epsilon,
        "gate_threshold_unchanged": gate_threshold,
        "initial_backend": initial_backend,
        "model_eval_audit": model_audit,
        "profiles": summaries,
        "interpretation_note": (
            "Deterministic algorithms ensure repeatability for a fixed operation shape; "
            "they do not guarantee bitwise invariance across different batch shapes."
        ),
    }
    write_csv(output / "per_sample_comparisons.csv", rows_out)
    write_json(output / "summary.json", report)
    print(json.dumps({"status": "completed", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
