from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from edge_finer_research.evaluation.metrics import image_metrics

from .checkpoints import save_checkpoint_atomic
from .efficientnet_arcface import EfficientNetArcFace


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def build_model(config: dict[str, Any], *, pretrained: bool = True) -> EfficientNetArcFace:
    values = config["model"]
    return EfficientNetArcFace(
        num_classes=len(config["experiment"]["class_names"]),
        embedding_dim=int(values["embedding_dim"]),
        dropout_before_embedding=float(values["dropout_before_embedding"]),
        dropout_after_embedding=float(values["dropout_after_embedding"]),
        arc_scale=float(values["arcface_scale"]),
        arc_margin=float(values["arcface_margin"]),
        arc_easy_margin=bool(values["arcface_easy_margin"]),
        feature_mode=str(values["feature_mode"]),
        pretrained=pretrained,
    )


@torch.no_grad()
def validate(model: EfficientNetArcFace, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    fake_scores: list[float] = []
    losses: list[float] = []
    criterion = nn.CrossEntropyLoss()
    for images, targets, _sample_ids in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model.forward_logits(images)
        losses.append(float(criterion(logits, targets).item()))
        probability = logits.softmax(dim=1)
        labels.extend(targets.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        fake_scores.extend(probability[:, 1].cpu().tolist())
    measured = image_metrics(labels, predictions, fake_scores)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "image_f1": measured.paper_f1,
        "positive_f1": measured.positive_class_f1,
        "auc": measured.auc,
        "sensitivity": measured.sensitivity,
        "specificity": measured.specificity,
    }


def train_experiment1(
    model: EfficientNetArcFace,
    train_loader,
    validation_loader,
    config: dict[str, Any],
    output_dir: str | Path,
    device: torch.device,
) -> None:
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    schedule = config["scheduler"]
    if schedule["kind"] != "cosine":
        raise ValueError("Only the declared cosine reconstruction schedule is implemented")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training["epochs"]),
        eta_min=float(schedule["minimum_learning_rate"]),
    )
    criterion = nn.CrossEntropyLoss()
    use_amp = bool(training["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.to(device)
    best_f1 = -1.0
    output = Path(output_dir)
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        for images, labels, _sample_ids in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model.forward_train(images, labels)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
        scheduler.step()
        validation = validate(model, validation_loader, device)
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_loss": running_loss / max(len(train_loader), 1),
            "validation": validation,
            "config": config,
            "inference_policy": "label_free_scaled_cosine",
        }
        if epoch % int(training["checkpoint_every"]) == 0:
            save_checkpoint_atomic(state, output / "checkpoints" / f"epoch_{epoch:03d}.pt")
        if validation["image_f1"] > best_f1:
            best_f1 = validation["image_f1"]
            save_checkpoint_atomic(state, output / "checkpoints" / "best_validation_f1.pt")
