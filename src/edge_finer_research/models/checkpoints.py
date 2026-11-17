from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import torch


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint_atomic(state: dict[str, Any], destination: str | Path) -> str:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)
    digest = checkpoint_sha256(path)
    hash_temporary = path.with_suffix(path.suffix + ".sha256.tmp")
    hash_destination = path.with_suffix(path.suffix + ".sha256")
    hash_temporary.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    os.replace(hash_temporary, hash_destination)
    return digest


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("Checkpoint must be a mapping containing model_state")
    return checkpoint
