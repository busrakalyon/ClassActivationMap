from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from edge_finer_research.cams.edgecam import PaperEdgeCAM
from edge_finer_research.cams.computation_profile import (
    apply_scientific_cam_profile,
    verify_scientific_cam_runtime,
)
from edge_finer_research.config import load_toml
from edge_finer_research.data.dataset import ManifestClassificationDataset, build_transforms
from edge_finer_research.data.manifest import read_manifest
from edge_finer_research.models.checkpoints import load_checkpoint
from edge_finer_research.models.training import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    config = load_toml(args.config)
    rows = [row for row in read_manifest(args.manifest) if row.split == args.split]
    if not rows:
        raise RuntimeError(f"No manifest rows for split={args.split}")
    device = torch.device(args.device)
    computation_profile = apply_scientific_cam_profile(
        device, batch_size=args.batch_size
    )
    model = build_model(config, pretrained=False).to(device=device, dtype=torch.float32)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    dataset = ManifestClassificationDataset(rows, args.image_root, build_transforms(config, train=False))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    output = args.output_dir / args.checkpoint.stem
    output.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, object]] = []
    runtime_verification: dict[str, object] | None = None
    target_layer = model.resolve_target_layer(str(config["model"]["target_layer"]))
    image_size = int(config["experiment"]["image_size"])
    edgecam_config = config["edgecam"]
    with PaperEdgeCAM(
        model,
        target_layer,
        epsilon=float(edgecam_config["epsilon"]),
        apply_relu=bool(edgecam_config["final_relu"]),
        normalize=bool(edgecam_config["normalize"]),
        output_size=(image_size, image_size),
    ) as extractor:
        for images, labels, sample_ids in loader:
            batch_images = images.to(device=device, dtype=torch.float32)
            runtime_verification = verify_scientific_cam_runtime(model, batch_images)
            with torch.amp.autocast(device_type="cuda", enabled=False):
                result = extractor.extract(
                    batch_images, int(config["experiment"]["positive_class"])
                )
            probabilities = result.logits.softmax(dim=1).cpu().numpy()
            for index, sample_id in enumerate(sample_ids):
                predicted = int(result.predicted_classes[index].item())
                valid_for_fusion = float(probabilities[index, 1]) >= float(
                    edgecam_config["classification_gate"]
                )
                np.savez_compressed(
                    output / f"{sample_id}.npz",
                    cam=result.maps[index].cpu().numpy().astype(np.float32),
                    logits=result.logits[index].cpu().numpy().astype(np.float32),
                    predicted_class=predicted,
                    valid_for_epoch_fusion=valid_for_fusion,
                )
                predictions.append(
                    {
                        "sample_id": sample_id,
                        "label": int(labels[index]),
                        "predicted_label": predicted,
                        "fake_score": float(probabilities[index, 1]),
                    }
                )
    with (output / "image_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)
    if runtime_verification is None:
        raise RuntimeError("No CAM batches were processed")
    (output / "cam_run_metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "split": args.split,
                "samples": len(rows),
                "computation_profile": computation_profile,
                "runtime_verification": runtime_verification,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
