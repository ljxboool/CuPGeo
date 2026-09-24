#!/usr/bin/env python3
"""Build leakage-checked source-only protocols for the five-centre Fundus DG data.

Input manifests must first be produced by ``prepare_fundus_dg_5centre.py``.
This tool deliberately keeps target-domain rows out of both train and
source-validation manifests.  It supports a strict single-source experiment
and a multi-source leave-one-domain-out experiment with the same interface.

The original public train/test partitions are used as follows by default:

* each source domain's ``train`` partition -> source training;
* each source domain's ``test`` partition -> source-only checkpoint selection;
* target domain's ``test`` partition -> sealed target evaluation.

The benchmark contains segmentation labels but no glaucoma labels, so all
output manifests remain segmentation-only and must be trained with
``cls_weight: 0``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


FIELDS = (
    "image",
    "mask",
    "glaucoma",
    "patient_id",
    "device",
    "split",
    "image_id",
    "mask_encoding",
    "source_domain",
)
VALID_SPLITS = frozenset(("train", "test"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a source-only Fundus DG protocol without target-label leakage."
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("manifests/fundus_dg_5centre"),
        help="Directory containing <domain>_train.csv and <domain>_test.csv.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="One or more source acquisition domains, e.g. REFUGE ORIGA.",
    )
    parser.add_argument("--target", required=True, help="Held-out target acquisition domain.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Empty/new directory where source_train.csv, source_val.csv and target_eval.csv are written.",
    )
    parser.add_argument(
        "--source-train-splits",
        nargs="+",
        choices=sorted(VALID_SPLITS),
        default=["train"],
    )
    parser.add_argument(
        "--source-val-splits",
        nargs="+",
        choices=sorted(VALID_SPLITS),
        default=["test"],
    )
    parser.add_argument(
        "--target-eval-splits",
        nargs="+",
        choices=sorted(VALID_SPLITS),
        default=["test"],
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacement of existing generated protocol files.",
    )
    return parser.parse_args()


def _slug(domain: str) -> str:
    return domain.casefold().replace("-", "_")


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Domain manifest is missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        if fields != FIELDS:
            raise ValueError(f"Unexpected manifest schema in {path}: {fields}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Domain manifest is empty: {path}")
    return rows


def _domain_rows(manifest_dir: Path, domain: str, requested_splits: Iterable[str]) -> list[dict[str, str]]:
    requested = tuple(requested_splits)
    rows: list[dict[str, str]] = []
    for split in requested:
        path = manifest_dir / f"{_slug(domain)}_{split}.csv"
        loaded = _read_rows(path)
        if any(row["device"] != domain for row in loaded):
            raise ValueError(f"Device metadata in {path} does not match requested domain {domain}")
        if any(row["split"] != split for row in loaded):
            raise ValueError(f"Split metadata in {path} does not match requested split {split}")
        if any(row["glaucoma"] for row in loaded):
            raise ValueError(f"This segmentation-only benchmark unexpectedly has a glaucoma label in {path}")
        rows.extend(loaded)
    return rows


def _assert_unique(rows: list[dict[str, str]], name: str) -> None:
    ids = [row["image_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate image IDs in {name}")
    patients = [row["patient_id"] for row in rows]
    if len(patients) != len(set(patients)):
        raise ValueError(
            f"Duplicate patient IDs in {name}; the public benchmark uses image-ID proxies, so this is unexpected"
        )


def _assert_disjoint(left: list[dict[str, str]], right: list[dict[str, str]], left_name: str, right_name: str) -> None:
    left_ids = {row["image_id"] for row in left}
    right_ids = {row["image_id"] for row in right}
    overlap = sorted(left_ids & right_ids)
    if overlap:
        raise ValueError(f"Image leakage between {left_name} and {right_name}: {overlap[:5]}")
    left_patients = {row["patient_id"] for row in left}
    right_patients = {row["patient_id"] for row in right}
    patient_overlap = sorted(left_patients & right_patients)
    if patient_overlap:
        raise ValueError(
            f"Patient leakage between {left_name} and {right_name}: {patient_overlap[:5]}"
        )


def _write_csv(path: Path, rows: list[dict[str, str]], *, force: bool) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty protocol file: {path}")
    if path.exists() and not force:
        raise FileExistsError(f"Protocol file already exists: {path}; rerun with --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    manifest_dir = args.manifest_dir.resolve()
    output_dir = args.output_dir.resolve()
    sources = tuple(args.sources)
    target = str(args.target)
    if len(set(sources)) != len(sources):
        raise ValueError(f"Duplicate source domain: {sources}")
    if target in sources:
        raise ValueError("The held-out target domain may not also appear in --sources")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is non-empty: {output_dir}; use --force to replace protocol files")

    source_train = [
        row
        for domain in sources
        for row in _domain_rows(manifest_dir, domain, args.source_train_splits)
    ]
    source_val = [
        row
        for domain in sources
        for row in _domain_rows(manifest_dir, domain, args.source_val_splits)
    ]
    target_eval = _domain_rows(manifest_dir, target, args.target_eval_splits)

    for name, rows in (
        ("source_train", source_train),
        ("source_val", source_val),
        ("target_eval", target_eval),
    ):
        _assert_unique(rows, name)
    _assert_disjoint(source_train, source_val, "source_train", "source_val")
    _assert_disjoint(source_train, target_eval, "source_train", "target_eval")
    _assert_disjoint(source_val, target_eval, "source_val", "target_eval")

    _write_csv(output_dir / "source_train.csv", source_train, force=args.force)
    _write_csv(output_dir / "source_val.csv", source_val, force=args.force)
    _write_csv(output_dir / "target_eval.csv", target_eval, force=args.force)

    metadata = {
        "protocol": "strict_source_only",
        "sources": list(sources),
        "target": target,
        "source_train_splits": list(args.source_train_splits),
        "source_val_splits": list(args.source_val_splits),
        "target_eval_splits": list(args.target_eval_splits),
        "target_used_for_model_selection": False,
        "classification_labels_available": False,
        "required_training_settings": {
            "cls_weight": 0.0,
            "selection_metric": "val_seg_dice",
        },
        "counts": {
            "source_train": len(source_train),
            "source_val": len(source_val),
            "target_eval": len(target_eval),
        },
    }
    metadata_path = output_dir / "protocol.json"
    if metadata_path.exists() and not args.force:
        raise FileExistsError(f"Protocol metadata already exists: {metadata_path}; rerun with --force to replace it")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
