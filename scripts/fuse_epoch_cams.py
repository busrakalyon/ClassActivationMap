from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from edge_finer_research.cams.epoch_fusion import fuse_epoch_cams
from edge_finer_research.data.manifest import read_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cam-dir", type=Path, action="append")
    source.add_argument("--cam-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--method", choices=("max", "mean"), default="max")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cam_directories = args.cam_dir or sorted(
        item for item in args.cam_root.iterdir() if item.is_dir()
    )
    if not cam_directories:
        raise RuntimeError("No checkpoint CAM directories found")
    for row in (item for item in read_manifest(args.manifest) if item.split == args.split):
        maps: list[np.ndarray] = []
        valid: list[bool] = []
        for directory in cam_directories:
            path = directory / f"{row.sample_id}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as archive:
                maps.append(archive["cam"])
                valid.append(bool(archive["valid_for_epoch_fusion"]))
        result = fuse_epoch_cams(maps, valid=valid, method=args.method)
        fused = result.map.astype(np.float32)
        if fused.shape != (row.height, row.width):
            fused = np.asarray(
                Image.fromarray(fused, mode="F").resize((row.width, row.height), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
        np.save(args.output_dir / f"{row.sample_id}.npy", fused)
        np.save(args.output_dir / f"{row.sample_id}__pre_sop.npy", fused >= args.threshold)
        (args.output_dir / f"{row.sample_id}.json").write_text(
            json.dumps(
                {"valid_epochs": result.valid_count, "total_epochs": result.total_count, "method": args.method},
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
