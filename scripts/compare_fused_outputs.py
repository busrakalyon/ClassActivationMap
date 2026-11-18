from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from edge_finer_research.data.manifest import read_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare materialized and streaming fusion outputs")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()
    rows = [row for row in read_manifest(args.manifest) if row.split == args.split]
    if not rows:
        raise RuntimeError(f"No manifest rows for split={args.split}")
    per_sample = []
    all_close = True
    metadata_equal = True
    for row in rows:
        reference = np.load(
            args.reference_dir / f"{row.sample_id}.npy", allow_pickle=False
        ).astype(np.float32, copy=False)
        candidate = np.load(
            args.candidate_dir / f"{row.sample_id}.npy", allow_pickle=False
        ).astype(np.float32, copy=False)
        if reference.shape != candidate.shape:
            raise RuntimeError(
                f"Shape mismatch for {row.sample_id}: {reference.shape} != {candidate.shape}"
            )
        difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
        close = bool(
            np.allclose(reference, candidate, atol=args.atol, rtol=args.rtol, equal_nan=False)
        )
        all_close &= close
        reference_meta = json.loads(
            (args.reference_dir / f"{row.sample_id}.json").read_text(encoding="utf-8")
        )
        candidate_meta = json.loads(
            (args.candidate_dir / f"{row.sample_id}.json").read_text(encoding="utf-8")
        )
        counts_equal = (
            reference_meta["valid_epochs"] == candidate_meta["valid_epochs"]
            and reference_meta["total_epochs"] == candidate_meta["total_epochs"]
            and reference_meta["method"] == candidate_meta["method"]
        )
        metadata_equal &= counts_equal
        per_sample.append(
            {
                "sample_id": row.sample_id,
                "allclose": close,
                "array_equal": bool(np.array_equal(reference, candidate)),
                "max_abs_difference": float(difference.max(initial=0.0)),
                "mean_abs_difference": float(difference.mean()),
                "valid_epoch_metadata_equal": counts_equal,
            }
        )
    report = {
        "status": "passed" if all_close and metadata_equal else "failed",
        "split": args.split,
        "samples": len(rows),
        "atol": args.atol,
        "rtol": args.rtol,
        "all_maps_close": all_close,
        "all_valid_epoch_metadata_equal": metadata_equal,
        "maximum_abs_difference": max(item["max_abs_difference"] for item in per_sample),
        "mean_of_sample_mean_abs_difference": float(
            np.mean([item["mean_abs_difference"] for item in per_sample])
        ),
        "per_sample": per_sample,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_sample"}, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
