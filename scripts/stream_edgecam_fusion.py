from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from edge_finer_research.cams.edgecam import PaperEdgeCAM
from edge_finer_research.cams.streaming_fusion import (
    initial_streaming_state,
    load_state,
    save_state_atomic,
    update_streaming_max,
    write_json_atomic,
    write_npy_atomic,
)
from edge_finer_research.config import load_toml
from edge_finer_research.data.dataset import ManifestClassificationDataset, build_transforms
from edge_finer_research.data.manifest import ManifestRow, read_manifest
from edge_finer_research.models.checkpoints import checkpoint_sha256, load_checkpoint
from edge_finer_research.models.training import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Disk-bounded, resumable EdgeCAM epoch max-fusion"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint-dir", type=Path)
    source.add_argument("--checkpoint", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.fullmatch(r"epoch_(\d+)", path.stem)
    return (int(match.group(1)), path.name) if match else (2**31 - 1, path.name)


def resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    paths = list(args.checkpoint or [])
    if args.checkpoint_dir is not None:
        paths = list(args.checkpoint_dir.glob("epoch_*.pt"))
    paths = sorted((path.resolve() for path in paths), key=checkpoint_sort_key)
    if not paths:
        raise RuntimeError("No epoch checkpoints found")
    if len(set(paths)) != len(paths):
        raise RuntimeError("Checkpoint list contains duplicates")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return paths


def state_path(state_dir: Path, sample_id: str) -> Path:
    token = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]
    return state_dir / f"{token}.npz"


def resize_native(cam: np.ndarray, row: ManifestRow) -> np.ndarray:
    values = np.asarray(cam, dtype=np.float32)
    if values.shape == (row.height, row.width):
        return values
    return np.asarray(
        Image.fromarray(values, mode="F").resize(
            (row.width, row.height), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_contract(
    args: argparse.Namespace,
    rows: list[ManifestRow],
    checkpoints: list[Path],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "cam_provider": "paper_edgecam",
        "provider_interface": "CAMBatch maps/logits/predicted_classes",
        "fusion": "gated_streaming_pixelwise_max",
        "dtype": "float32",
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "image_root": str(args.image_root.resolve()),
        "split": args.split,
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(row.sample_id for row in rows).encode("utf-8")
        ).hexdigest(),
        "samples": len(rows),
        "checkpoint_paths": [str(path) for path in checkpoints],
        "checkpoint_sha256": [checkpoint_sha256(path) for path in checkpoints],
        "total_checkpoints": len(checkpoints),
        "image_size": int(config["experiment"]["image_size"]),
        "positive_class": int(config["experiment"]["positive_class"]),
        "epsilon": float(config["edgecam"]["epsilon"]),
        "final_relu": bool(config["edgecam"]["final_relu"]),
        "normalize": bool(config["edgecam"]["normalize"]),
        "classification_gate": float(config["edgecam"]["classification_gate"]),
        "final_threshold": float(args.threshold),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "device": str(args.device),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def load_or_initialize_state(
    path: Path,
    row: ManifestRow,
    *,
    image_size: int,
    checkpoint_index: int,
) -> Any:
    if not path.is_file():
        if checkpoint_index != 0:
            raise RuntimeError(
                f"Missing state for {row.sample_id} before checkpoint index {checkpoint_index}"
            )
        return initial_streaming_state((image_size, image_size))
    state = load_state(path, expected_sample_id=row.sample_id)
    if state.map.shape != (image_size, image_size):
        raise RuntimeError(f"State shape mismatch for {row.sample_id}: {state.map.shape}")
    return state


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size must be positive and num workers cannot be negative")
    if not 0.0 < args.threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    config = load_toml(args.config)
    rows = [row for row in read_manifest(args.manifest) if row.split == args.split]
    if not rows:
        raise RuntimeError(f"No manifest rows for split={args.split}")
    if len({row.sample_id for row in rows}) != len(rows):
        raise RuntimeError("Selected manifest split contains duplicate sample IDs")
    checkpoints = resolve_checkpoints(args)
    contract = run_contract(args, rows, checkpoints, config)
    output = args.output_dir
    state_dir = output / ".streaming_state"
    metadata_path = state_dir / "run_contract.json"
    completed_path = output / "completed.json"

    if args.resume:
        if not metadata_path.is_file():
            raise RuntimeError("--resume requested but no streaming state contract exists")
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise RuntimeError("Resume contract differs from config/manifest/checkpoints/options")
        if completed_path.is_file():
            print(json.dumps({"status": "already_completed", "output": str(output)}, indent=2))
            return
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {output}; use a new directory or --resume"
            )
        state_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(metadata_path, contract)

    state_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    dataset = ManifestClassificationDataset(
        rows, args.image_root, build_transforms(config, train=False)
    )
    by_id = {row.sample_id: row for row in rows}
    index_by_id = {row.sample_id: index for index, row in enumerate(rows)}
    image_size = int(config["experiment"]["image_size"])
    target_class = int(config["experiment"]["positive_class"])
    gate_threshold = float(config["edgecam"]["classification_gate"])

    for checkpoint_index, checkpoint_path in enumerate(checkpoints):
        pending_indices: list[int] = []
        for row in rows:
            path = state_path(state_dir, row.sample_id)
            state = load_or_initialize_state(
                path, row, image_size=image_size, checkpoint_index=checkpoint_index
            )
            if state.processed_checkpoint_count == checkpoint_index:
                pending_indices.append(index_by_id[row.sample_id])
            elif state.processed_checkpoint_count == checkpoint_index + 1:
                continue
            elif state.processed_checkpoint_count > checkpoint_index + 1:
                continue
            else:
                raise RuntimeError(
                    f"State for {row.sample_id} is behind checkpoint index {checkpoint_index}"
                )
        if not pending_indices:
            write_json_atomic(
                state_dir / "progress.json",
                {
                    "completed_checkpoint_count": checkpoint_index + 1,
                    "total_checkpoints": len(checkpoints),
                    "checkpoint": checkpoint_path.name,
                    "status": "already_processed",
                },
            )
            continue

        model = build_model(config, pretrained=False).to(device)
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        target_layer = model.resolve_target_layer(str(config["model"]["target_layer"]))
        loader = DataLoader(
            Subset(dataset, pending_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        processed_now = 0
        edgecam = config["edgecam"]
        with PaperEdgeCAM(
            model,
            target_layer,
            epsilon=float(edgecam["epsilon"]),
            apply_relu=bool(edgecam["final_relu"]),
            normalize=bool(edgecam["normalize"]),
            output_size=(image_size, image_size),
        ) as extractor:
            for images, labels, sample_ids in loader:
                result = extractor.extract(images.to(device), target_class)
                probabilities = result.logits.softmax(dim=1).cpu().numpy()
                for item_index, sample_id in enumerate(sample_ids):
                    row = by_id[sample_id]
                    path = state_path(state_dir, sample_id)
                    state = load_or_initialize_state(
                        path,
                        row,
                        image_size=image_size,
                        checkpoint_index=checkpoint_index,
                    )
                    if state.processed_checkpoint_count != checkpoint_index:
                        raise RuntimeError(
                            f"Concurrent or inconsistent update detected for {sample_id}"
                        )
                    logits = result.logits[item_index].cpu().numpy().astype(np.float32)
                    fake_score = float(probabilities[item_index, target_class])
                    valid = fake_score >= gate_threshold
                    updated = update_streaming_max(
                        state,
                        result.maps[item_index].cpu().numpy().astype(np.float32),
                        valid=valid,
                        logits=logits,
                        predicted_class=int(result.predicted_classes[item_index]),
                        fake_score=fake_score,
                    )
                    save_state_atomic(path, updated, sample_id=sample_id)
                    if int(labels[item_index]) != row.label:
                        raise RuntimeError(f"Dataset label mismatch for {sample_id}")
                    processed_now += 1
                write_json_atomic(
                    state_dir / "progress.json",
                    {
                        "completed_checkpoint_count": checkpoint_index,
                        "total_checkpoints": len(checkpoints),
                        "current_checkpoint": checkpoint_path.name,
                        "processed_samples_in_current_checkpoint": processed_now,
                        "total_samples": len(rows),
                        "status": "in_progress",
                    },
                )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        write_json_atomic(
            state_dir / "progress.json",
            {
                "completed_checkpoint_count": checkpoint_index + 1,
                "total_checkpoints": len(checkpoints),
                "checkpoint": checkpoint_path.name,
                "status": "checkpoint_complete",
            },
        )

    predictions: list[dict[str, Any]] = []
    for row in rows:
        state = load_state(state_path(state_dir, row.sample_id), expected_sample_id=row.sample_id)
        if state.processed_checkpoint_count != len(checkpoints):
            raise RuntimeError(f"Incomplete state for {row.sample_id}")
        fused = resize_native(state.map, row)
        if fused.dtype != np.float32 or not np.isfinite(fused).all():
            raise RuntimeError(f"Invalid final CAM for {row.sample_id}")
        write_npy_atomic(output / f"{row.sample_id}.npy", fused)
        write_npy_atomic(output / f"{row.sample_id}__pre_sop.npy", fused >= args.threshold)
        write_json_atomic(
            output / f"{row.sample_id}.json",
            {
                "valid_epochs": state.valid_count,
                "total_epochs": len(checkpoints),
                "method": "max",
                "streaming": True,
                "dtype": "float32",
            },
        )
        predictions.append(
            {
                "sample_id": row.sample_id,
                "label": row.label,
                "predicted_label": state.predicted_class,
                "fake_score": state.fake_score,
            }
        )
    write_csv_atomic(output / "image_predictions.csv", predictions)
    write_json_atomic(
        completed_path,
        {
            "status": "completed",
            "samples": len(rows),
            "total_epochs": len(checkpoints),
            "last_checkpoint": checkpoints[-1].name,
            "fusion": "gated_streaming_pixelwise_max",
            "state_retained": True,
        },
    )
    print(json.dumps({"status": "completed", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
