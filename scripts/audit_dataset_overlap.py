from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from edge_finer_research.data.manifest import read_manifest, write_manifest
from edge_finer_research.data.overlap import (
    ImageFingerprint,
    exact_cross_matches,
    fingerprint_image,
    perceptual_cross_matches,
    source_cross_matches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit CASIAv2 train/validation against CASIAv1 test for leakage"
    )
    parser.add_argument("--casia-v2-manifest", type=Path, required=True)
    parser.add_argument("--casia-v2-image-root", type=Path, required=True)
    parser.add_argument("--casia-v1-manifest", type=Path, required=True)
    parser.add_argument("--casia-v1-image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phash-max-distance", type=int, default=8)
    return parser.parse_args()


def label_name(value: int) -> str:
    return "fake" if value == 1 else "real"


def pair_record(left: ImageFingerprint, right: ImageFingerprint) -> dict[str, Any]:
    return {
        "casia_v2_split": left.row.split,
        "casia_v2_sample_id": left.row.sample_id,
        "casia_v2_label": left.row.label,
        "casia_v2_class": label_name(left.row.label),
        "casia_v2_manipulation_type": left.row.manipulation_type,
        "casia_v2_image_relpath": left.row.image_relpath,
        "casia_v1_sample_id": right.row.sample_id,
        "casia_v1_label": right.row.label,
        "casia_v1_class": label_name(right.row.label),
        "casia_v1_manipulation_type": right.row.manipulation_type,
        "casia_v1_image_relpath": right.row.image_relpath,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def grouped_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        (
            str(row["casia_v2_split"]),
            str(row["casia_v2_class"]),
            str(row["casia_v1_class"]),
        )
        for row in records
    )
    return {
        f"{split}__v2_{left_class}__v1_{right_class}": count
        for (split, left_class, right_class), count in sorted(counts.items())
    }


def main() -> None:
    args = parse_args()
    if not 0 <= args.phash_max_distance <= 64:
        raise ValueError("--phash-max-distance must be between 0 and 64")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    v2_rows = [
        row for row in read_manifest(args.casia_v2_manifest) if row.split in {"train", "val"}
    ]
    v1_rows = [row for row in read_manifest(args.casia_v1_manifest) if row.split == "test"]
    if not v2_rows or not v1_rows:
        raise RuntimeError("Expected non-empty CASIAv2 train/val and CASIAv1 test rows")
    if len({row.sample_id for row in [*v2_rows, *v1_rows]}) != len(v2_rows) + len(v1_rows):
        raise RuntimeError("Sample IDs are not globally unique across the selected manifests")

    v2_fingerprints = [fingerprint_image(row, args.casia_v2_image_root) for row in v2_rows]
    v1_fingerprints = [fingerprint_image(row, args.casia_v1_image_root) for row in v1_rows]

    raw_pairs = exact_cross_matches(v2_fingerprints, v1_fingerprints, attribute="file_sha256")
    rgb_pairs = exact_cross_matches(v2_fingerprints, v1_fingerprints, attribute="rgb_sha256")
    phash_pairs = perceptual_cross_matches(
        v2_fingerprints, v1_fingerprints, max_distance=args.phash_max_distance
    )
    source_pairs = source_cross_matches(v2_fingerprints, v1_fingerprints)

    exact_pair_ids = {
        (left.row.sample_id, right.row.sample_id) for left, right in [*raw_pairs, *rgb_pairs]
    }

    raw_records = [
        {**pair_record(left, right), "file_sha256": left.file_sha256}
        for left, right in raw_pairs
    ]
    rgb_records = [
        {**pair_record(left, right), "rgb_sha256": left.rgb_sha256}
        for left, right in rgb_pairs
    ]
    phash_records = [
        {
            **pair_record(left, right),
            "hamming_distance": distance,
            "casia_v2_phash": f"{left.perceptual_hash:016x}",
            "casia_v1_phash": f"{right.perceptual_hash:016x}",
            "review_status": (
                "confirmed_by_exact_hash"
                if (left.row.sample_id, right.row.sample_id) in exact_pair_ids
                else "manual_review_required"
            ),
        }
        for left, right, distance in phash_pairs
    ]
    source_records = [
        {
            **pair_record(left, right),
            "shared_source_ids": ";".join(shared),
            "relationship": (
                "same_source_different_manipulation_candidate"
                if left.row.sample_id != right.row.sample_id
                else "same_sample_id"
            ),
        }
        for left, right, shared in source_pairs
    ]

    base_fields = list(pair_record(v2_fingerprints[0], v1_fingerprints[0]))
    write_csv(args.output_dir / "exact_file_sha256.csv", raw_records, [*base_fields, "file_sha256"])
    write_csv(args.output_dir / "exact_decoded_rgb.csv", rgb_records, [*base_fields, "rgb_sha256"])
    write_csv(
        args.output_dir / "perceptual_hash_candidates.csv",
        phash_records,
        [
            *base_fields,
            "hamming_distance",
            "casia_v2_phash",
            "casia_v1_phash",
            "review_status",
        ],
    )
    write_csv(
        args.output_dir / "shared_source_ids.csv",
        source_records,
        [*base_fields, "shared_source_ids", "relationship"],
    )

    exclusions: dict[str, dict[str, Any]] = {}

    def add_exclusion(left: ImageFingerprint, right: ImageFingerprint, reason: str) -> None:
        item = exclusions.setdefault(
            right.row.sample_id,
            {
                "sample_id": right.row.sample_id,
                "label": right.row.label,
                "class": label_name(right.row.label),
                "image_relpath": right.row.image_relpath,
                "recommended_action": "exclude_from_source_disjoint_test",
                "reasons": set(),
                "matching_casia_v2_samples": set(),
                "matching_casia_v2_splits": set(),
            },
        )
        item["reasons"].add(reason)
        item["matching_casia_v2_samples"].add(left.row.sample_id)
        item["matching_casia_v2_splits"].add(left.row.split)

    for left, right in raw_pairs:
        add_exclusion(left, right, "exact_file_sha256")
    for left, right in rgb_pairs:
        add_exclusion(left, right, "exact_decoded_rgb")
    for left, right, _shared in source_pairs:
        add_exclusion(left, right, "shared_casia_source_id")

    exclusion_rows: list[dict[str, Any]] = []
    for item in sorted(exclusions.values(), key=lambda value: str(value["sample_id"])):
        exclusion_rows.append(
            {
                **{key: value for key, value in item.items() if not isinstance(value, set)},
                "reasons": ";".join(sorted(item["reasons"])),
                "matching_casia_v2_samples": ";".join(sorted(item["matching_casia_v2_samples"])),
                "matching_casia_v2_splits": ";".join(sorted(item["matching_casia_v2_splits"])),
            }
        )
    exclusion_fields = [
        "sample_id",
        "label",
        "class",
        "image_relpath",
        "recommended_action",
        "reasons",
        "matching_casia_v2_samples",
        "matching_casia_v2_splits",
    ]
    write_csv(args.output_dir / "recommended_test_exclusions.csv", exclusion_rows, exclusion_fields)

    excluded_ids = set(exclusions)
    if excluded_ids:
        write_manifest(
            [row for row in v1_rows if row.sample_id not in excluded_ids],
            args.output_dir / "casia_v1_source_disjoint_proposed.csv",
        )

    unique_phash_review = (
        {right.row.sample_id for _left, right, _distance in phash_pairs} - excluded_ids
    )
    unique_by_split: dict[str, dict[str, int]] = defaultdict(dict)
    for split in ("train", "val"):
        for name, pairs in (
            ("exact_file_sha256", raw_pairs),
            ("exact_decoded_rgb", rgb_pairs),
            ("shared_source_ids", source_pairs),
            ("perceptual_hash_candidates", phash_pairs),
        ):
            unique_by_split[split][name] = len(
                {right.row.sample_id for left, right, *_rest in pairs if left.row.split == split}
            )

    summary = {
        "protocol": {
            "casia_v2_manifest": str(args.casia_v2_manifest.resolve()),
            "casia_v1_manifest": str(args.casia_v1_manifest.resolve()),
            "comparisons": ["train-test", "validation-test"],
            "file_hash": "SHA256 of encoded file bytes",
            "decoded_hash": "SHA256 of RGB dimensions and decoded RGB bytes",
            "perceptual_hash": "64-bit DCT pHash over 32x32 luminance",
            "phash_max_hamming_distance": args.phash_max_distance,
            "phash_interpretation": "candidate_only_manual_review_required",
            "source_identity": "intersection of CASIA IDs parsed/stored in manifest source_ids",
        },
        "input_samples": {"casia_v2_train_val": len(v2_rows), "casia_v1_test": len(v1_rows)},
        "pair_counts": {
            "exact_file_sha256": len(raw_records),
            "exact_decoded_rgb": len(rgb_records),
            "perceptual_hash_candidates": len(phash_records),
            "shared_source_ids": len(source_records),
        },
        "pair_counts_by_split_and_class": {
            "exact_file_sha256": grouped_counts(raw_records),
            "exact_decoded_rgb": grouped_counts(rgb_records),
            "perceptual_hash_candidates": grouped_counts(phash_records),
            "shared_source_ids": grouped_counts(source_records),
        },
        "unique_test_matches_by_casia_v2_split": unique_by_split,
        "recommended_test_exclusions": {
            "count": len(excluded_ids),
            "real": sum(item["label"] == 0 for item in exclusions.values()),
            "fake": sum(item["label"] == 1 for item in exclusions.values()),
            "proposed_manifest": (
                "casia_v1_source_disjoint_proposed.csv" if excluded_ids else None
            ),
        },
        "perceptual_only_manual_review_candidates": len(unique_phash_review),
        "important_note": (
            "The proposed clean manifest excludes exact byte/RGB matches and shared CASIA "
            "source IDs. "
            "Perceptual-hash-only candidates are not excluded automatically."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8", errors="strict"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
