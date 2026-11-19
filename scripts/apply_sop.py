from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from edge_finer_research.cams.streaming_fusion import write_json_atomic, write_npy_atomic
from edge_finer_research.data.manifest import read_manifest
from edge_finer_research.proposals.mcg import MCGNpzProposalProvider
from edge_finer_research.proposals.sop import NpzProposalProvider, select_object_proposal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply EdgeCAM paper Eq. 7-8 SOP scoring to cached proposals"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fused-cam-dir", type=Path, required=True)
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--proposal-metadata-dir", type=Path)
    parser.add_argument(
        "--proposal-format", choices=("mcg_hwk", "legacy_nhw"), default="mcg_hwk"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--expected-count", type=int, default=50)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_completed_result(
    mask_path: Path, json_path: Path, shape: tuple[int, int]
) -> dict[str, Any]:
    if not mask_path.is_file() or not json_path.is_file():
        raise RuntimeError(f"Partial SOP result exists: {mask_path.stem}")
    mask = np.load(mask_path, allow_pickle=False)
    if mask.shape != shape or mask.dtype != np.bool_:
        raise ValueError(f"Invalid existing SOP mask: {mask_path}")
    return json.loads(json_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if args.expected_count != 50:
        raise ValueError("EdgeCAM SOP reconstruction requires exactly 50 proposals")
    if args.proposal_format == "mcg_hwk" and args.proposal_metadata_dir is None:
        raise ValueError("--proposal-metadata-dir is required for mcg_hwk caches")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed_path = args.output_dir / "completed.json"
    index_path = args.output_dir / "selection_index.csv"

    rows = [row for row in read_manifest(args.manifest) if row.split == args.split]
    if not rows:
        raise RuntimeError(f"No manifest rows for split={args.split}")
    if len({row.sample_id for row in rows}) != len(rows):
        raise RuntimeError("Selected manifest split contains duplicate sample IDs")

    mcg_provider = (
        MCGNpzProposalProvider(args.proposal_dir, expected_count=args.expected_count)
        if args.proposal_format == "mcg_hwk"
        else None
    )
    legacy_provider = (
        NpzProposalProvider(args.proposal_dir, expected_count=args.expected_count)
        if args.proposal_format == "legacy_nhw"
        else None
    )

    selection_rows: list[dict[str, Any]] = []
    for row in rows:
        cam_path = args.fused_cam_dir / f"{row.sample_id}.npy"
        if not cam_path.is_file():
            raise FileNotFoundError(cam_path)
        cam = np.load(cam_path, allow_pickle=False)
        if cam.dtype != np.float32 or cam.shape != (row.height, row.width):
            raise ValueError(
                f"Expected native float32 CAM {(row.height, row.width)}, "
                f"found {cam.dtype} {cam.shape}: {cam_path}"
            )
        if not np.isfinite(cam).all():
            raise ValueError(f"CAM contains NaN/Inf: {cam_path}")

        mask_path = args.output_dir / f"{row.sample_id}__post_sop.npy"
        result_path = args.output_dir / f"{row.sample_id}.json"
        if mask_path.exists() or result_path.exists():
            if not args.resume:
                raise FileExistsError(mask_path if mask_path.exists() else result_path)
            item = read_completed_result(mask_path, result_path, cam.shape)
            selection_rows.append(item)
            continue

        if args.proposal_format == "mcg_hwk":
            proposal_metadata_path = args.proposal_metadata_dir / f"{row.sample_id}.json"
            if not proposal_metadata_path.is_file():
                raise FileNotFoundError(proposal_metadata_path)
            proposal_metadata = json.loads(
                proposal_metadata_path.read_text(encoding="utf-8")
            )
            expected_metadata = {
                "sample_id": row.sample_id,
                "image_height": row.height,
                "image_width": row.width,
                "saved_top_k": args.expected_count,
                "mcg_mode": "accurate",
            }
            differences = {
                key: (proposal_metadata.get(key), value)
                for key, value in expected_metadata.items()
                if proposal_metadata.get(key) != value
            }
            if differences:
                raise ValueError(
                    f"MCG proposal metadata mismatch for {row.sample_id}: {differences}"
                )

        proposals = (
            mcg_provider.get_for_shape(row.sample_id, height=row.height, width=row.width)
            if mcg_provider is not None
            else legacy_provider.get(row.sample_id)  # type: ignore[union-attr]
        )
        result = select_object_proposal(
            cam,
            proposals,
            expected_count=args.expected_count,
            kernel_size=args.kernel_size,
            iterations=args.iterations,
        )
        item = {
            "sample_id": row.sample_id,
            "proposal_index_zero_based": result.proposal_index,
            "proposal_rank_one_based": result.proposal_index + 1,
            "selected_score": result.score,
            "all_scores": result.scores.tolist(),
            "proposal_format": args.proposal_format,
            "kernel_size": args.kernel_size,
            "morphology_iterations": args.iterations,
            "cam_dtype": "float32",
            "ground_truth_used_for_selection": False,
            "image_label_used_for_selection": False,
        }
        write_npy_atomic(mask_path, result.mask.astype(bool, copy=False))
        write_json_atomic(result_path, item)
        selection_rows.append(item)

    if completed_path.exists() or index_path.exists():
        if not args.resume:
            raise FileExistsError(completed_path if completed_path.exists() else index_path)
        if not completed_path.is_file() or not index_path.is_file():
            raise RuntimeError("Partial SOP completion metadata exists")
        print(completed_path.read_text(encoding="utf-8"))
        return

    with index_path.open("x", newline="", encoding="utf-8") as handle:
        fields = [key for key in selection_rows[0] if key != "all_scores"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selection_rows)
    summary = {
        "status": "completed",
        "samples": len(rows),
        "split": args.split,
        "proposal_format": args.proposal_format,
        "proposal_metadata_dir": (
            str(args.proposal_metadata_dir.resolve())
            if args.proposal_metadata_dir is not None
            else None
        ),
        "expected_count": args.expected_count,
        "kernel_size": args.kernel_size,
        "morphology_iterations": args.iterations,
        "scoring": "paper_eq_7_8_sum(proposal_boundary * continuous_fused_cam)",
        "ground_truth_used_for_selection": False,
        "image_label_used_for_selection": False,
    }
    write_json_atomic(completed_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
