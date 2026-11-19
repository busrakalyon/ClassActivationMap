from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from edge_finer_research.data.manifest import ManifestRow, read_manifest, sha256_file
from edge_finer_research.proposals.mcg import (
    load_mcg_cache,
    mcg_git_commit,
    parse_matlab_intermediate,
    validate_reusable_cache,
    write_json_atomic,
    write_mcg_cache_atomic,
)


@dataclass(frozen=True)
class PendingSample:
    row: ManifestRow
    image_path: Path
    image_sha256: str
    height: int
    width: int
    cache_path: Path
    metadata_path: Path
    intermediate_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reusable official-MCG reconstructed top-k proposal caches"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--mcg-root", type=Path, required=True)
    parser.add_argument("--matlab-bin", type=Path, required=True)
    parser.add_argument(
        "--matlab-script-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "matlab",
    )
    parser.add_argument("--mode", choices=("accurate",), default="accurate")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")
    return parser.parse_args()


def utc_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def matlab_quote(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def inspect_sample(
    row: ManifestRow,
    *,
    image_root: Path,
    output_dir: Path,
    metadata_dir: Path,
    work_dir: Path,
) -> PendingSample:
    image_path = (image_root / row.image_relpath).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as image:
        width, height = image.size
    if row.height and row.width and (height, width) != (row.height, row.width):
        raise ValueError(
            f"Manifest/image dimensions differ for {row.sample_id}: "
            f"{(row.height, row.width)} vs {(height, width)}"
        )
    return PendingSample(
        row=row,
        image_path=image_path,
        image_sha256=sha256_file(image_path),
        height=height,
        width=width,
        cache_path=output_dir / f"{row.sample_id}.npz",
        metadata_path=metadata_dir / f"{row.sample_id}.json",
        intermediate_dir=work_dir / row.sample_id,
    )


def metadata_payload(
    sample: PendingSample,
    matlab_metadata: dict[str, Any],
    *,
    mode: str,
    top_k: int,
    mcg_root: Path,
    mcg_commit: str | None,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "generator": "official_mcg_reconstructed",
        "sample_id": sample.row.sample_id,
        "dataset": sample.row.dataset,
        "split": sample.row.split,
        "image_path": str(sample.image_path),
        "image_sha256": sample.image_sha256,
        "image_height": sample.height,
        "image_width": sample.width,
        "proposal_count_total": int(matlab_metadata["proposal_count_total"]),
        "saved_top_k": top_k,
        "mcg_mode": mode,
        "mcg_source_path": str(mcg_root.resolve()),
        "mcg_source_commit": mcg_commit,
        "matlab_version": str(matlab_metadata["matlab_version"]),
        "proposal_masks_path": str(sample.cache_path.resolve()),
        "npz_layout": "H,W,K",
        "npz_mask_dtype": "uint8_binary",
        "npz_keys": ["masks", "scores", "ranks"],
        "rank_semantics": "one_based_official_mcg_descending_score_order",
    }


def finalize_sample(
    sample: PendingSample,
    *,
    mode: str,
    top_k: int,
    mcg_root: Path,
    mcg_commit: str | None,
) -> None:
    cache, matlab_metadata = parse_matlab_intermediate(
        sample.intermediate_dir,
        expected_height=sample.height,
        expected_width=sample.width,
        expected_count=top_k,
    )
    if str(matlab_metadata["sample_id"]) != sample.row.sample_id:
        raise ValueError(f"MATLAB sample ID mismatch for {sample.row.sample_id}")
    if str(matlab_metadata["mcg_mode"]) != mode:
        raise ValueError(f"MATLAB MCG mode mismatch for {sample.row.sample_id}")

    if sample.cache_path.exists():
        existing = load_mcg_cache(
            sample.cache_path,
            expected_height=sample.height,
            expected_width=sample.width,
            expected_count=top_k,
        )
        if not (
            np.array_equal(existing.masks_hwk, cache.masks_hwk)
            and np.array_equal(existing.scores, cache.scores)
            and np.array_equal(existing.ranks, cache.ranks)
        ):
            raise ValueError(f"Existing partial cache differs for {sample.row.sample_id}")
    else:
        write_mcg_cache_atomic(sample.cache_path, cache)

    payload = metadata_payload(
        sample,
        matlab_metadata,
        mode=mode,
        top_k=top_k,
        mcg_root=mcg_root,
        mcg_commit=mcg_commit,
    )
    if sample.metadata_path.exists():
        existing_metadata = json.loads(sample.metadata_path.read_text(encoding="utf-8"))
        if existing_metadata != payload:
            raise ValueError(f"Existing metadata differs for {sample.row.sample_id}")
    else:
        write_json_atomic(sample.metadata_path, payload)

    if not validate_reusable_cache(
        sample.cache_path,
        sample.metadata_path,
        sample_id=sample.row.sample_id,
        image_sha256=sample.image_sha256,
        expected_height=sample.height,
        expected_width=sample.width,
        expected_count=top_k,
        mode=mode,
        mcg_source_commit=mcg_commit,
    ):
        raise RuntimeError(f"Finalized cache did not validate: {sample.row.sample_id}")


def safe_remove_intermediate(path: Path, work_root: Path) -> None:
    resolved, root = path.resolve(), work_root.resolve()
    if resolved.parent != root:
        raise RuntimeError(f"Refusing to remove intermediate outside work root: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)


def write_job(
    path: Path,
    samples: list[PendingSample],
    *,
    mcg_root: Path,
    mode: str,
    top_k: int,
) -> None:
    payload = {
        "mcg_root": str(mcg_root.resolve()),
        "mode": mode,
        "top_k": top_k,
        "items": [
            {
                "sample_id": sample.row.sample_id,
                "image_path": str(sample.image_path),
                "intermediate_dir": str(sample.intermediate_dir.resolve()),
            }
            for sample in samples
        ],
    }
    write_json_atomic(path, payload)


def main() -> None:
    args = parse_args()
    if args.top_k != 50:
        raise ValueError("EdgeCAM MCG reconstruction requires --top-k 50")
    if args.chunk_size < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("chunk-size and limit must be positive")
    if not args.mcg_root.is_dir():
        raise NotADirectoryError(args.mcg_root)
    if not args.matlab_bin.is_file():
        raise FileNotFoundError(args.matlab_bin)
    if not (args.matlab_script_dir / "generate_mcg_topk_batch.m").is_file():
        raise FileNotFoundError(args.matlab_script_dir / "generate_mcg_topk_batch.m")

    rows = [row for row in read_manifest(args.manifest) if row.split == args.split]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"No manifest rows for split={args.split}")
    if len({row.sample_id for row in rows}) != len(rows):
        raise RuntimeError("Selected manifest split contains duplicate sample IDs")

    output_dir = args.output_dir.resolve()
    metadata_dir = args.metadata_dir.resolve()
    work_dir = (args.work_dir or (output_dir / ".mcg_work")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    run_token = utc_token()
    mcg_commit = mcg_git_commit(args.mcg_root)

    samples = [
        inspect_sample(
            row,
            image_root=args.image_root,
            output_dir=output_dir,
            metadata_dir=metadata_dir,
            work_dir=work_dir,
        )
        for row in rows
    ]
    pending: list[PendingSample] = []
    skipped = 0
    for sample in samples:
        cache_exists = sample.cache_path.is_file()
        metadata_exists = sample.metadata_path.is_file()
        intermediate_complete = (sample.intermediate_dir / "completed.json").is_file()
        if cache_exists != metadata_exists:
            if not intermediate_complete:
                raise RuntimeError(
                    f"Partial cache without recoverable MATLAB intermediate: {sample.row.sample_id}"
                )
            finalize_sample(
                sample,
                mode=args.mode,
                top_k=args.top_k,
                mcg_root=args.mcg_root,
                mcg_commit=mcg_commit,
            )
        if validate_reusable_cache(
            sample.cache_path,
            sample.metadata_path,
            sample_id=sample.row.sample_id,
            image_sha256=sample.image_sha256,
            expected_height=sample.height,
            expected_width=sample.width,
            expected_count=args.top_k,
            mode=args.mode,
            mcg_source_commit=mcg_commit,
        ):
            skipped += 1
            if not args.keep_intermediate:
                safe_remove_intermediate(sample.intermediate_dir, work_dir)
            continue
        if intermediate_complete:
            finalize_sample(
                sample,
                mode=args.mode,
                top_k=args.top_k,
                mcg_root=args.mcg_root,
                mcg_commit=mcg_commit,
            )
            skipped += 1
            if not args.keep_intermediate:
                safe_remove_intermediate(sample.intermediate_dir, work_dir)
            continue
        pending.append(sample)

    if args.verify_only and pending:
        raise RuntimeError(f"MCG cache is incomplete: {len(pending)} samples missing")
    if args.verify_only:
        print(json.dumps({"status": "verified", "samples": len(samples)}, indent=2))
        return

    failures: list[dict[str, Any]] = []
    created = 0
    for chunk_index, start in enumerate(range(0, len(pending), args.chunk_size), start=1):
        chunk = pending[start : start + args.chunk_size]
        job_path = work_dir / f"job_{run_token}_{chunk_index:04d}.json"
        write_job(
            job_path,
            chunk,
            mcg_root=args.mcg_root,
            mode=args.mode,
            top_k=args.top_k,
        )
        expression = (
            f"addpath('{matlab_quote(args.matlab_script_dir)}'); "
            f"generate_mcg_topk_batch('{matlab_quote(job_path)}')"
        )
        process = subprocess.run(
            [str(args.matlab_bin.resolve()), "-batch", expression], check=False
        )
        for sample in chunk:
            try:
                finalize_sample(
                    sample,
                    mode=args.mode,
                    top_k=args.top_k,
                    mcg_root=args.mcg_root,
                    mcg_commit=mcg_commit,
                )
                created += 1
                if not args.keep_intermediate:
                    safe_remove_intermediate(sample.intermediate_dir, work_dir)
            except Exception as exception:
                matlab_error_path = sample.intermediate_dir / "error.json"
                matlab_error = (
                    json.loads(matlab_error_path.read_text(encoding="utf-8"))
                    if matlab_error_path.is_file()
                    else None
                )
                failures.append(
                    {
                        "sample_id": sample.row.sample_id,
                        "error": str(exception),
                        "matlab_error": matlab_error,
                        "matlab_return_code": process.returncode,
                    }
                )

    summary = {
        "status": "failed" if failures else "completed",
        "manifest": str(args.manifest.resolve()),
        "split": args.split,
        "samples": len(samples),
        "created": created,
        "reused": skipped,
        "failed": len(failures),
        "top_k": args.top_k,
        "mcg_mode": args.mode,
        "mcg_root": str(args.mcg_root.resolve()),
        "mcg_commit": mcg_commit,
        "output_dir": str(output_dir),
        "metadata_dir": str(metadata_dir),
    }
    summary_path = metadata_dir / f"run_{run_token}.json"
    write_json_atomic(summary_path, summary)
    if failures:
        error_path = metadata_dir / f"errors_{run_token}.jsonl"
        with error_path.open("x", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure) + "\n")
        raise RuntimeError(f"MCG failed for {len(failures)} samples; see {error_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
