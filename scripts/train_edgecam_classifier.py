from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from edge_finer_research.config import load_toml
from edge_finer_research.data.dataset import ManifestClassificationDataset, build_transforms
from edge_finer_research.data.manifest import read_manifest, validate_manifest
from edge_finer_research.models.training import build_model, seed_everything, train_experiment1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument(
        "--mask-root",
        type=Path,
        required=True,
        help="Used only for manifest integrity checks; masks are not classifier targets",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_toml(args.config)
    rows = read_manifest(args.manifest)
    issues = validate_manifest(
        rows,
        image_root=args.image_root,
        mask_root=args.mask_root,
    )
    if any(issue.level == "error" for issue in issues):
        raise RuntimeError(f"Manifest has errors: {issues[:10]}")
    train_rows = [row for row in rows if row.split == "train"]
    validation_rows = [row for row in rows if row.split == "val"]
    if not train_rows or not validation_rows:
        raise RuntimeError("Both train and val rows are required")
    seed_everything(int(config["experiment"]["seed"]))
    training = config["training"]
    common = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training["num_workers"]),
        "pin_memory": args.device.startswith("cuda"),
    }
    train_loader = DataLoader(
        # Weak supervision: this dataset reads images and image-level labels only.
        ManifestClassificationDataset(train_rows, args.image_root, build_transforms(config, train=True)),
        shuffle=True,
        **common,
    )
    validation_loader = DataLoader(
        ManifestClassificationDataset(validation_rows, args.image_root, build_transforms(config, train=False)),
        shuffle=False,
        **common,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, args.output_dir / "resolved_experiment_config.toml")
    shutil.copy2(args.manifest, args.output_dir / "training_manifest.csv")
    manifest_digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    metadata = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "manifest_sha256": manifest_digest,
        "mask_usage": "manifest_validation_only_not_classifier_target",
        "seed": int(config["experiment"]["seed"]),
        "inference_policy": "label_free_scaled_cosine",
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    train_experiment1(build_model(config), train_loader, validation_loader, config, args.output_dir, device)


if __name__ == "__main__":
    main()
