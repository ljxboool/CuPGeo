#!/usr/bin/env python3
"""Prepare the public five-centre OD/OC domain-generalization benchmark.

Source archive: Zenodo record 8009107, ``Processed_Fundus_Images.zip``.

The archive intentionally retains the original mask encoding shared by all five
domains: 255=background, 128=optic-disc rim, and 0=optic cup.  This project
expects either two binary masks or one canonical three-class mask with
0=background, 1=disc (excluding cup), 2=cup.  This script creates the latter
without changing the extracted source files.

For the two RIGA domains, the CSV names a base TIFF mask while the archive
contains six expert annotations named ``<stem>-1.tif`` through ``-6.tif``.
The official TriD implementation used for this benchmark resolves that path to
``-1.tif``.  We reproduce that documented convention exactly and record it in
the preparation report; it is not an arbitrary expert selection.

The benchmark has OD/OC masks only.  Generated manifests retain an empty
``glaucoma`` column to satisfy the common manifest schema, so a segmentation-
only training configuration must set ``cls_weight: 0`` and select by
``val_seg_dice``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


DOMAINS = ("REFUGE", "Drishti_GS", "ORIGA", "BinRushed", "Magrabia")
SPLITS = ("train", "test")
EXPECTED_VALUES = frozenset((0, 128, 255))
MANIFEST_FIELDS = (
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


@dataclass(frozen=True)
class SourceRecord:
    domain: str
    split: str
    image_path: Path
    raw_mask_path: Path
    image_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the Zenodo five-centre Fundus DG archive into canonical masks and manifests."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/benchmarks/fundus_dg_5centre"),
        help="Directory containing extracted/ and where prepared/ will be created.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("manifests/fundus_dg_5centre"),
        help="Directory for one train/test manifest per acquisition domain.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Audit all source rows without writing masks, manifests, or a report.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacement if an existing generated canonical mask disagrees with the source conversion.",
    )
    return parser.parse_args()


def _canonical_id(domain: str, image_relpath: str) -> str:
    rel = Path(image_relpath)
    pieces = (domain, *rel.with_suffix("").parts)
    return "__".join(piece.replace(" ", "_") for piece in pieces)


def _resolve_riga_expert_one(mask_path: Path) -> Path:
    if mask_path.is_file():
        return mask_path
    if mask_path.suffix.lower() == ".tif":
        expert_one = mask_path.with_name(f"{mask_path.stem}-1{mask_path.suffix}")
        if expert_one.is_file():
            return expert_one
    raise FileNotFoundError(
        "Mask path does not exist. For RIGA, expected the official TriD expert-1 "
        f"annotation beside the CSV base path: {mask_path}"
    )


def read_records(extracted_root: Path) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    seen_ids: set[str] = set()
    for domain in DOMAINS:
        for split in SPLITS:
            csv_path = extracted_root / f"{domain}_{split}.csv"
            if not csv_path.is_file():
                raise FileNotFoundError(f"Official split CSV not found: {csv_path}")
            with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if set(reader.fieldnames or ()) != {"image", "mask"}:
                    raise ValueError(f"Unexpected CSV schema in {csv_path}: {reader.fieldnames}")
                for line_no, row in enumerate(reader, start=2):
                    image_rel = row.get("image", "")
                    mask_rel = row.get("mask", "")
                    if not image_rel or not mask_rel:
                        raise ValueError(f"{csv_path}:{line_no} has an empty image or mask path")
                    image_path = extracted_root / image_rel
                    if not image_path.is_file():
                        raise FileNotFoundError(f"Image listed by {csv_path}:{line_no} is missing: {image_path}")
                    raw_mask_path = _resolve_riga_expert_one(extracted_root / mask_rel)
                    image_id = _canonical_id(domain, image_rel)
                    if image_id in seen_ids:
                        raise ValueError(f"Duplicate canonical image ID: {image_id}")
                    seen_ids.add(image_id)
                    records.append(
                        SourceRecord(
                            domain=domain,
                            split=split,
                            image_path=image_path,
                            raw_mask_path=raw_mask_path,
                            image_id=image_id,
                        )
                    )
    return records


def canonicalize_mask(mask: Image.Image, *, source: Path) -> np.ndarray:
    raw = np.asarray(mask.convert("L"))
    values = set(np.unique(raw).tolist())
    unknown = sorted(values - EXPECTED_VALUES)
    if unknown:
        raise ValueError(f"Unexpected mask values {unknown} in {source}")
    # Official encoding: 255 background, 128 disc rim, 0 cup.
    canonical = np.zeros(raw.shape, dtype=np.uint8)
    canonical[raw < 255] = 1
    canonical[raw == 0] = 2
    return canonical


def write_png_if_needed(destination: Path, canonical: np.ndarray, *, force: bool) -> bool:
    """Write one canonical mask atomically. Return True when a file was written."""
    if destination.is_file():
        with Image.open(destination) as existing:
            existing_arr = np.asarray(existing.convert("L"))
        if np.array_equal(existing_arr, canonical):
            return False
        if not force:
            raise FileExistsError(
                f"Existing generated mask differs from source conversion: {destination}. "
                "Refuse to overwrite; inspect it or rerun with --force."
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    Image.fromarray(canonical, mode="L").save(temporary, format="PNG")
    temporary.replace(destination)
    return True


def relative_to_dataset(path: Path, dataset_root: Path) -> str:
    try:
        return path.relative_to(dataset_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path must be inside --dataset-root: {path}") from exc


def manifest_row(record: SourceRecord, dataset_root: Path, canonical_path: Path) -> dict[str, str]:
    image_rel = relative_to_dataset(record.image_path, dataset_root)
    mask_rel = relative_to_dataset(canonical_path, dataset_root)
    return {
        "image": image_rel,
        "mask": mask_rel,
        "glaucoma": "",
        # The public benchmark ships no patient identifiers.  The image ID is a
        # conservative proxy that prevents a single image appearing in both splits.
        "patient_id": record.image_id,
        "device": record.domain,
        "split": record.split,
        "image_id": record.image_id,
        "mask_encoding": "three_class_012",
        "source_domain": record.domain,
    }


def write_manifest(rows: Iterable[dict[str, str]], path: Path) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Refusing to write an empty manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(materialized)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    extracted_root = dataset_root / "extracted"
    prepared_root = dataset_root / "prepared"
    manifest_dir = args.manifest_dir.resolve()
    if not extracted_root.is_dir():
        raise FileNotFoundError(
            f"Expected an extracted official archive at {extracted_root}; run unzip first."
        )

    records = read_records(extracted_root)
    counts: dict[str, Counter[str]] = {domain: Counter() for domain in DOMAINS}
    rows_by_domain_split: dict[tuple[str, str], list[dict[str, str]]] = {
        (domain, split): [] for domain in DOMAINS for split in SPLITS
    }
    raw_value_counts: Counter[int] = Counter()
    created = 0

    for record in records:
        with Image.open(record.image_path) as image, Image.open(record.raw_mask_path) as raw_mask:
            if image.size != raw_mask.size:
                raise ValueError(
                    f"Image/mask size mismatch for {record.image_id}: "
                    f"{image.size} vs {raw_mask.size}"
                )
            source_arr = np.asarray(raw_mask.convert("L"))
            raw_value_counts.update(np.unique(source_arr).tolist())
            canonical = canonicalize_mask(raw_mask, source=record.raw_mask_path)

        output_path = prepared_root / "masks" / record.domain / f"{record.image_id}.png"
        if not args.check_only:
            created += int(write_png_if_needed(output_path, canonical, force=args.force))
            rows_by_domain_split[(record.domain, record.split)].append(
                manifest_row(record, dataset_root, output_path)
            )
        counts[record.domain][record.split] += 1

    expected_counts = {
        "REFUGE": {"train": 320, "test": 80},
        "Drishti_GS": {"train": 50, "test": 51},
        "ORIGA": {"train": 500, "test": 150},
        "BinRushed": {"train": 156, "test": 39},
        "Magrabia": {"train": 76, "test": 19},
    }
    actual_counts = {domain: dict(counter) for domain, counter in counts.items()}
    if actual_counts != expected_counts:
        raise ValueError(f"Unexpected split counts: {actual_counts}")

    if not args.check_only:
        for domain in DOMAINS:
            for split in SPLITS:
                write_manifest(
                    rows_by_domain_split[(domain, split)],
                    manifest_dir / f"{domain.lower()}_{split}.csv",
                )
        report = {
            "dataset": "A Fundus Image Dataset for Domain Generalization in Joint Segmentation of Optic Disc and Optic Cup",
            "zenodo_record": "8009107",
            "source_archive": "raw/Processed_Fundus_Images.zip",
            "source_archive_md5": "a28baa241c45c95d14f9279e4c664d9a",
            "domains": actual_counts,
            "raw_mask_encoding": {"0": "optic cup", "128": "optic-disc rim", "255": "background"},
            "canonical_mask_encoding": {
                "0": "background",
                "1": "optic-disc region excluding cup",
                "2": "optic cup; the loader forms the OD channel from classes 1 and 2",
            },
            "riga_policy": "Use expert-1 (-1.tif), matching the official TriD public loader.",
            "classification_labels": "not provided; glaucoma fields in manifests are intentionally blank",
            "patient_id_policy": "image_id proxy; public archive has no patient metadata",
            "masks_created_this_run": created,
        }
        prepared_root.mkdir(parents=True, exist_ok=True)
        report_path = prepared_root / "preparation_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    mode = "audit only" if args.check_only else "prepared"
    print(f"{mode}: {len(records)} images across {len(DOMAINS)} domains")
    print("split counts:", json.dumps(actual_counts, sort_keys=True))
    print("raw mask values:", dict(sorted(raw_value_counts.items())))
    if not args.check_only:
        print(f"canonical masks written this run: {created}")
        print(f"manifests: {manifest_dir}")


if __name__ == "__main__":
    main()
