from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from edge_finer_research.data.manifest import read_manifest
from edge_finer_research.proposals.sop import NpzProposalProvider, select_object_proposal


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply only the verified SOP scoring/selection stage")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fused-cam-dir", type=Path, required=True)
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--expected-count", type=int, default=50)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provider = NpzProposalProvider(args.proposal_dir, expected_count=args.expected_count)
    for row in (item for item in read_manifest(args.manifest) if item.split == args.split):
        cam_path = args.fused_cam_dir / f"{row.sample_id}.npy"
        if not cam_path.is_file():
            raise FileNotFoundError(cam_path)
        result = select_object_proposal(
            np.load(cam_path, allow_pickle=False),
            provider.get(row.sample_id),
            expected_count=args.expected_count,
            kernel_size=args.kernel_size,
            iterations=args.iterations,
        )
        np.save(args.output_dir / f"{row.sample_id}__post_sop.npy", result.mask)
        (args.output_dir / f"{row.sample_id}.json").write_text(
            json.dumps(
                {"proposal_index": result.proposal_index, "score": result.score}, indent=2
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
