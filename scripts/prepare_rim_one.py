#!/usr/bin/env python3
"""Prepare the official RIM-ONE DL hospital partition for C3-TTA.

Only ``partitioned_by_hospital`` is used.  Its training set (HUC) becomes the
source domain, while its test set (the published HUMS/HCSC aggregate) becomes
the target domain.  The target adaptation CSV is deliberately label-free.

RIM-ONE DL does not publish patient identifiers.  Consequently, the source
validation split is stratified at image level and ``patient_id`` is populated
with ``image_id`` solely to satisfy the common manifest schema.  The generated
metadata records this limitation; the split must not be described as
patient-disjoint in a paper.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from c3tta.data.manifest import validate_manifest, write_manifest


SOURCE_DOMAIN = "HUC"
TARGET_DOMAIN = "HUMS_HCSC"
OFFICIAL_COUNTS = {
    ("training_set", 0): 195,
    ("training_set", 1): 116,
    ("test_set", 0): 118,
    ("test_set", 1): 56,
}
MASK_RE = re.compile(
    r"^(?P<image_id>.+)-1-(?P<structure>Disc|Cup)-T\.png$", re.IGNORECASE
)


@dataclass(frozen=True)
class RimOneRecord:
    image_id: str
    image_path: Path
    od_mask_path: Path
    oc_mask_path: Path
    glaucoma: int
    official_split: str
    domain: str


def find_dataset_roots(path: str | Path) -> tuple[Path, Path]:
    """Find the image and reference-segmentation roots below an extraction root."""

    requested = Path(path).expanduser().resolve()
    if not requested.is_dir():
        raise FileNotFoundError(f"RIM-ONE DL extraction directory does not exist: {requested}")

    hospital_dirs = sorted(
        directory.resolve()
        for directory in requested.rglob("partitioned_by_hospital")
        if directory.is_dir()
    )
    if requested.name == "partitioned_by_hospital":
        hospital_dirs = [requested]
    hospital_dirs = sorted(set(hospital_dirs))
    if len(hospital_dirs) != 1:
        raise ValueError(
            "Expected exactly one partitioned_by_hospital directory below "
            f"{requested}; found {len(hospital_dirs)}: {[str(p) for p in hospital_dirs]}"
        )
    image_root = hospital_dirs[0].parent

    segmentation_roots = {
        mask.parent.parent.resolve()
        for mask in requested.rglob("*-1-Disc-T.png")
        if mask.parent.name.casefold() in {"normal", "glaucoma"}
    }
    if len(segmentation_roots) != 1:
        raise ValueError(
            "Expected exactly one RIM-ONE DL reference-segmentation directory below "
            f"{requested}; found {len(segmentation_roots)}: "
            f"{[str(p) for p in sorted(segmentation_roots)]}"
        )
    return image_root, segmentation_roots.pop()


def _discover_images(image_root: Path) -> dict[str, tuple[Path, int, str, str]]:
    records: dict[str, tuple[Path, int, str, str]] = {}
    hospital_root = image_root / "partitioned_by_hospital"
    for official_split, domain in (
        ("training_set", SOURCE_DOMAIN),
        ("test_set", TARGET_DOMAIN),
    ):
        for class_name, glaucoma in (("normal", 0), ("glaucoma", 1)):
            directory = hospital_root / official_split / class_name
            if not directory.is_dir():
                raise FileNotFoundError(f"Missing official RIM-ONE DL directory: {directory}")
            paths = sorted(
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() == ".png"
            )
            if not paths:
                raise ValueError(f"No PNG images found in {directory}")
            for path in paths:
                image_id = path.stem
                if image_id in records:
                    raise ValueError(
                        f"Duplicate image basename {image_id}: {records[image_id][0]}, {path}"
                    )
                records[image_id] = (path, glaucoma, official_split, domain)
    return records


def _discover_masks(
    segmentation_root: Path,
) -> tuple[dict[str, dict[str, Path]], dict[str, int], int]:
    masks: dict[str, dict[str, Path]] = {}
    mask_labels: dict[str, int] = {}
    ignored_txt = 0
    for class_name, glaucoma in (("normal", 0), ("glaucoma", 1)):
        directory = segmentation_root / class_name
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing reference-segmentation directory: {directory}")
        ignored_txt += sum(path.is_file() for path in directory.glob("*.txt"))
        for path in sorted(directory.glob("*.png")):
            match = MASK_RE.fullmatch(path.name)
            if match is None:
                raise ValueError(f"Unexpected PNG in reference segmentations: {path}")
            image_id = match.group("image_id")
            structure = match.group("structure").casefold()
            current_label = mask_labels.setdefault(image_id, glaucoma)
            if current_label != glaucoma:
                raise ValueError(f"Masks for {image_id} occur in both diagnosis directories")
            image_masks = masks.setdefault(image_id, {})
            if structure in image_masks:
                raise ValueError(
                    f"Duplicate {structure} mask for {image_id}: "
                    f"{image_masks[structure]}, {path}"
                )
            image_masks[structure] = path
    return masks, mask_labels, ignored_txt


def _validate_inventory(
    images: dict[str, tuple[Path, int, str, str]],
    masks: dict[str, dict[str, Path]],
    mask_labels: dict[str, int],
    image_root: Path,
    allow_subset: bool,
) -> tuple[list[RimOneRecord], dict[str, object]]:
    image_ids = set(images)
    mask_ids = set(masks)
    if image_ids != mask_ids:
        raise ValueError(
            "Image/mask inventory mismatch: "
            f"images_without_masks={sorted(image_ids - mask_ids)[:10]}, "
            f"masks_without_images={sorted(mask_ids - image_ids)[:10]}"
        )

    observed_counts = Counter((split, label) for _, label, split, _ in images.values())
    if not allow_subset and dict(observed_counts) != OFFICIAL_COUNTS:
        readable = {
            f"{split}:{'glaucoma' if label else 'normal'}": count
            for (split, label), count in sorted(observed_counts.items())
        }
        raise ValueError(
            f"Official RIM-ONE DL inventory mismatch: expected {OFFICIAL_COUNTS}, "
            f"observed {readable}. Use --allow-subset only for a fixture/debug subset."
        )
    license_path = image_root / "LICENSE.txt"
    if not allow_subset and not license_path.is_file():
        raise FileNotFoundError(f"Official RIM-ONE DL LICENSE.txt is missing: {license_path}")

    records: list[RimOneRecord] = []
    cup_outside_disc: dict[str, int] = {}
    image_sizes: Counter[str] = Counter()
    for image_id in sorted(images):
        image_path, glaucoma, official_split, domain = images[image_id]
        if mask_labels[image_id] != glaucoma:
            raise ValueError(
                f"Diagnosis-directory mismatch for {image_id}: image={glaucoma}, "
                f"masks={mask_labels[image_id]}"
            )
        image_masks = masks[image_id]
        missing = {"disc", "cup"} - set(image_masks)
        if missing:
            raise FileNotFoundError(
                f"Missing {sorted(missing)} reference mask(s) for image {image_id}"
            )

        with Image.open(image_path) as image:
            image.load()
            size = image.size
        arrays: dict[str, np.ndarray] = {}
        for structure in ("disc", "cup"):
            mask_path = image_masks[structure]
            with Image.open(mask_path) as mask:
                mask.load()
                if mask.size != size:
                    raise ValueError(
                        f"Size mismatch for {image_id}: image={size}, "
                        f"{structure} mask={mask.size}"
                    )
                array = np.asarray(mask.convert("L"))
            values = set(int(value) for value in np.unique(array))
            if not values <= {0, 1, 255} or not any(value > 0 for value in values):
                raise ValueError(
                    f"{structure} mask for {image_id} must be non-empty binary; values={values}"
                )
            arrays[structure] = array > 0
        cup_outside_disc[image_id] = int(
            np.logical_and(arrays["cup"], ~arrays["disc"]).sum()
        )
        image_sizes[f"{size[0]}x{size[1]}"] += 1
        records.append(
            RimOneRecord(
                image_id=image_id,
                image_path=image_path,
                od_mask_path=image_masks["disc"],
                oc_mask_path=image_masks["cup"],
                glaucoma=glaucoma,
                official_split=official_split,
                domain=domain,
            )
        )

    qa: dict[str, object] = {
        "image_sizes": dict(sorted(image_sizes.items())),
        "images_with_cup_pixels_outside_disc": sum(
            count > 0 for count in cup_outside_disc.values()
        ),
        "total_cup_pixels_outside_disc": sum(cup_outside_disc.values()),
        "max_cup_pixels_outside_disc": max(cup_outside_disc.values(), default=0),
        "cup_was_intersected_with_disc": False,
        "mask_processing": "none; official PNG files are copied/linked byte-for-byte",
    }
    return records, qa


def stratified_source_split(
    records: Iterable[RimOneRecord], val_fraction: float, seed: int
) -> tuple[list[RimOneRecord], list[RimOneRecord]]:
    """Return a deterministic image-level split stratified by diagnosis."""

    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")
    source = [record for record in records if record.official_split == "training_set"]
    by_label = {label: [] for label in (0, 1)}
    for record in source:
        by_label[record.glaucoma].append(record)
    rng = random.Random(seed)
    validation_ids: set[str] = set()
    for label in (0, 1):
        group = sorted(by_label[label], key=lambda record: record.image_id)
        if len(group) < 2:
            raise ValueError(
                f"Stratified source split needs at least two class-{label} images; "
                f"found {len(group)}"
            )
        rng.shuffle(group)
        count = min(len(group) - 1, max(1, round(len(group) * val_fraction)))
        validation_ids.update(record.image_id for record in group[:count])
    train = sorted(
        (record for record in source if record.image_id not in validation_ids),
        key=lambda record: record.image_id,
    )
    validation = sorted(
        (record for record in source if record.image_id in validation_ids),
        key=lambda record: record.image_id,
    )
    return train, validation


def _relative_paths(record: RimOneRecord) -> tuple[str, str, str]:
    group = "source" if record.official_split == "training_set" else "target"
    image = f"images/{group}/{record.image_id}.png"
    od_mask = f"masks/{group}/{record.image_id}_od.png"
    oc_mask = f"masks/{group}/{record.image_id}_oc.png"
    return image, od_mask, oc_mask


def _labeled_row(record: RimOneRecord, split: str) -> dict[str, object]:
    image, od_mask, oc_mask = _relative_paths(record)
    return {
        "image": image,
        "od_mask": od_mask,
        "oc_mask": oc_mask,
        "glaucoma": record.glaucoma,
        # RIM-ONE DL publishes no patient IDs. This is a schema surrogate only.
        "patient_id": record.image_id,
        "device": record.domain,
        "split": split,
        "image_id": record.image_id,
        "mask_encoding": "separate_binary",
    }


def _adapt_row(record: RimOneRecord) -> dict[str, object]:
    image, _, _ = _relative_paths(record)
    return {
        "image": image,
        "patient_id": record.image_id,
        "device": record.domain,
        "split": "adapt",
        "image_id": record.image_id,
    }


def _place_file(source: Path, destination: Path, mode: str, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {destination}; pass --overwrite")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    try:
        if mode == "copy":
            shutil.copy2(source, temporary)
        elif mode == "hardlink":
            try:
                os.link(source, temporary)
            except OSError as exc:
                raise OSError(
                    f"Cannot hardlink {source} to {destination}; use --file-mode copy "
                    "when source/output are on different filesystems"
                ) from exc
        elif mode == "symlink":
            temporary.symlink_to(os.path.relpath(source.resolve(), destination.parent.resolve()))
        else:
            raise ValueError(f"Unsupported file mode: {mode}")
        temporary.replace(destination)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _validate_outputs(manifests: dict[str, Path], output_root: Path) -> None:
    modes = {"train": "source", "val": "source", "adapt": "target_adapt", "eval": "source"}
    errors: list[str] = []
    for name, path in manifests.items():
        for error in validate_manifest(path, modes[name], True, output_root):
            errors.append(f"{path}: {error}")
    if errors:
        raise RuntimeError("Generated manifest validation failed:\n" + "\n".join(errors))


def prepare_rim_one(
    archive_root: str | Path,
    output_root: str | Path,
    manifest_dir: str | Path = "manifests",
    *,
    val_fraction: float = 0.15,
    seed: int = 42,
    file_mode: str = "copy",
    overwrite: bool = False,
    allow_subset: bool = False,
) -> dict[str, object]:
    """Prepare RIM-ONE DL and return the metadata written beside its manifests."""

    if file_mode not in {"copy", "hardlink", "symlink"}:
        raise ValueError("file_mode must be copy, hardlink, or symlink")
    image_root, segmentation_root = find_dataset_roots(archive_root)
    images = _discover_images(image_root)
    masks, mask_labels, ignored_txt = _discover_masks(segmentation_root)
    records, mask_qa = _validate_inventory(
        images, masks, mask_labels, image_root, allow_subset
    )
    source_train, source_val = stratified_source_split(records, val_fraction, seed)
    target = sorted(
        (record for record in records if record.official_split == "test_set"),
        key=lambda record: record.image_id,
    )
    if not target:
        raise ValueError("Official test_set is empty")
    if {record.image_id for record in source_train + source_val} & {
        record.image_id for record in target
    }:
        raise ValueError("Source/target image ID overlap in official hospital partition")

    output_root = Path(output_root).expanduser().resolve()
    manifest_dir = Path(manifest_dir).expanduser().resolve()
    manifests = {
        name: manifest_dir / f"rim_one_{name}.csv"
        for name in ("train", "val", "adapt", "eval")
    }
    metadata_path = manifest_dir / "rim_one_prepare_metadata.json"
    outputs = [*manifests.values(), metadata_path]
    for record in records:
        image, od_mask, oc_mask = _relative_paths(record)
        outputs.extend([output_root / image, output_root / od_mask, output_root / oc_mask])
    existing = [path for path in outputs if path.exists() or path.is_symlink()]
    if existing and not overwrite:
        preview = ", ".join(str(path) for path in existing[:5])
        raise FileExistsError(f"Output(s) already exist; pass --overwrite: {preview}")

    for record in records:
        image, od_mask, oc_mask = _relative_paths(record)
        _place_file(record.image_path, output_root / image, file_mode, overwrite)
        _place_file(record.od_mask_path, output_root / od_mask, file_mode, overwrite)
        _place_file(record.oc_mask_path, output_root / oc_mask, file_mode, overwrite)

    rows = {
        "train": [_labeled_row(record, "train") for record in source_train],
        "val": [_labeled_row(record, "val") for record in source_val],
        "adapt": [_adapt_row(record) for record in target],
        "eval": [_labeled_row(record, "eval") for record in target],
    }
    if [row["image_id"] for row in rows["adapt"]] != [
        row["image_id"] for row in rows["eval"]
    ]:
        raise AssertionError("Internal error: target adaptation/evaluation order mismatch")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for name, path in manifests.items():
        write_manifest(rows[name], path)
    _validate_outputs(manifests, output_root)

    def label_counts(items: Iterable[RimOneRecord]) -> dict[str, int]:
        counts = Counter(record.glaucoma for record in items)
        return {"normal": counts[0], "glaucoma": counts[1]}

    metadata: dict[str, object] = {
        "dataset": "RIM-ONE DL",
        "official_partition": "partitioned_by_hospital",
        "resolved_image_root": str(image_root),
        "resolved_segmentation_root": str(segmentation_root),
        "prepared_data_root": str(output_root),
        "file_mode": file_mode,
        "protocol": {
            "source": "official training_set (HUC)",
            "target": "official test_set (HUMS/HCSC aggregate)",
            "target_adaptation_is_label_free": True,
            "target_adapt_eval_order_identical": True,
            "source_validation": "image-level diagnosis-stratified subset of HUC only",
            "source_validation_seed": seed,
            "source_validation_fraction_requested": val_fraction,
            "target_was_not_used_for_source_validation": True,
        },
        "patient_identity": {
            "published_patient_ids_available": False,
            "manifest_patient_id": "image_id surrogate",
            "patient_disjointness_verifiable": False,
            "required_reporting_limitation": (
                "RIM-ONE DL does not publish patient identifiers; source train/validation "
                "are image-disjoint, but patient-disjointness cannot be established."
            ),
        },
        "usage_restrictions_from_license": {
            "allowed_purposes": ["research", "education"],
            "copy_or_redistribution_permitted": False,
            "commercial_use_permitted": False,
            "license_path": str(image_root / "LICENSE.txt"),
            "do_not_publish_prepared_data": True,
            "official_repository": "https://github.com/miag-ull/rim-one-dl",
            "required_citation_doi": "10.5566/ias.2346",
        },
        "counts": {
            "all_images": len(records),
            "source_official_training_set": len(source_train) + len(source_val),
            "source_train": len(source_train),
            "source_val": len(source_val),
            "target_official_test_set": len(target),
            "target_adapt_rows": len(rows["adapt"]),
            "target_eval_rows": len(rows["eval"]),
            "source_official_by_label": label_counts(source_train + source_val),
            "source_train_by_label": label_counts(source_train),
            "source_val_by_label": label_counts(source_val),
            "target_by_label": label_counts(target),
            "official_png_masks": len(masks) * 2,
            "ignored_reference_txt_files": ignored_txt,
        },
        "mask_qa": mask_qa,
        "manifests": {name: str(path) for name, path in manifests.items()},
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare official RIM-ONE DL hospital-partition manifests."
    )
    parser.add_argument(
        "--archive-root",
        required=True,
        help="Common extraction root containing images and reference segmentations.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest-dir", default="manifests")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--file-mode", choices=("copy", "hardlink", "symlink"), default="copy"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Disable exact official-count checks for fixtures/debugging only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare_rim_one(
        archive_root=args.archive_root,
        output_root=args.output_root,
        manifest_dir=args.manifest_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        file_mode=args.file_mode,
        overwrite=args.overwrite,
        allow_subset=args.allow_subset,
    )
    print(json.dumps({"status": "ok", **metadata["counts"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
