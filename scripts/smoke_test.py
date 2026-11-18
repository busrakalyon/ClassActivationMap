from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from edge_finer_research.cams.edgecam import PaperEdgeCAM
from edge_finer_research.cams.epoch_fusion import fuse_epoch_cams
from edge_finer_research.evaluation.metrics import image_metrics, pixel_metrics
from edge_finer_research.models.checkpoints import load_checkpoint, save_checkpoint_atomic
from edge_finer_research.models.efficientnet_arcface import EfficientNetArcFace
from edge_finer_research.testing import run_deterministic_cam_control


def synthetic_batch(image_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    generator = torch.Generator().manual_seed(1729)
    images = torch.rand((4, 3, image_size, image_size), generator=generator) * 0.15
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    masks = np.zeros((4, image_size, image_size), dtype=bool)
    start, end = image_size // 4, 3 * image_size // 4
    images[labels == 1, :, start:end, start:end] += 0.8
    masks[labels.numpy() == 1, start:end, start:end] = True
    images = (images.clamp(0, 1) - torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
    images = images / torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return images.to(device), labels.to(device), masks


def build_model() -> EfficientNetArcFace:
    return EfficientNetArcFace(
        num_classes=2,
        embedding_dim=512,
        arc_scale=50.0,
        arc_margin=0.5,
        feature_mode="legacy_imagenet_logits",
        pretrained=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Data-free end-to-end Experiment-1 smoke test")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=Path("results/smoke"))
    args = parser.parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(1729)
    images, labels, masks = synthetic_batch(args.image_size, device)
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    checkpoints: list[Path] = []
    losses: list[float] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model.forward_train(images, labels), labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        checkpoint = output / "checkpoints" / f"epoch_{epoch:03d}.pt"
        save_checkpoint_atomic(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "smoke_test": True,
            },
            checkpoint,
        )
        checkpoints.append(checkpoint)

    maps_by_sample: list[list[np.ndarray]] = [[] for _ in range(len(images))]
    valid_by_sample: list[list[bool]] = [[] for _ in range(len(images))]
    final_logits: torch.Tensor | None = None
    raw_nonzero_cam_count = 0
    for checkpoint in checkpoints:
        restored = build_model().to(device)
        restored.load_state_dict(load_checkpoint(checkpoint, map_location=device)["model_state"], strict=True)
        layer = restored.resolve_target_layer("backbone.features.2")
        with PaperEdgeCAM(restored, layer, output_size=(args.image_size, args.image_size)) as extractor:
            result = extractor.extract(images, 1)
        final_logits = result.logits
        probabilities = result.logits.softmax(dim=1)[:, 1]
        if not torch.isfinite(result.maps).all():
            raise AssertionError("EdgeCAM produced non-finite values")
        raw_nonzero_cam_count += int(torch.count_nonzero(result.maps).item() > 0)
        for index in range(len(images)):
            maps_by_sample[index].append(result.maps[index].cpu().numpy())
            valid_by_sample[index].append(bool(probabilities[index].item() >= 0.5))

    assert final_logits is not None
    pixel_results: list[dict[str, float | int]] = []
    for index, (maps, valid) in enumerate(zip(maps_by_sample, valid_by_sample, strict=True)):
        fused = fuse_epoch_cams(maps, valid=valid, method="max")
        prediction = fused.map >= 0.5
        measured = pixel_metrics(masks[index], prediction)
        np.save(output / f"sample_{index}_fused.npy", fused.map.astype(np.float32))
        pixel_results.append(
            {
                "sample": index,
                "valid_epochs": fused.valid_count,
                "pixel_f1": measured.f1,
                "iou": measured.iou,
            }
        )
    scores = final_logits.softmax(dim=1)[:, 1].detach().cpu().numpy()
    predicted = final_logits.argmax(dim=1).detach().cpu().numpy()
    measured_image = image_metrics(labels.cpu().numpy(), predicted, scores)
    deterministic_control = run_deterministic_cam_control(device)
    report = {
        "status": "passed",
        "device": str(device),
        "torch": torch.__version__,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "losses": losses,
        "checkpoints": [path.name for path in checkpoints],
        "checkpoint_reload": True,
        "edgecam_finite": True,
        "raw_nonzero_cam_batches": raw_nonzero_cam_count,
        "image_f1": measured_image.paper_f1,
        "auc": measured_image.auc,
        "pixel_results": pixel_results,
        "deterministic_cam_control": deterministic_control,
        "note": "Synthetic smoke metrics are pipeline diagnostics, not research results.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
