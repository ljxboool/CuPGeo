#!/usr/bin/env python3
"""Prepare the official PAPILA archive for C3-TTA.

PAPILA stores optic-disc and optic-cup annotations as polygon vertices.  Each
text file contains ``x y`` coordinates; they must therefore be passed to an
image rasterizer in that order (array indexing would use ``[y, x]``).  This
script selects one explicitly named expert and preserves both polygons as
independent binary masks.  It never silently intersects the cup with the disc.

The official diagnosis is eye-specific: 0=healthy, 1=glaucoma, 2=suspicious.
The repository's classification head is binary, so suspicious eyes are either
excluded (default) or explicitly mapped to positive by ``--suspect-policy``.
The selected policy and mask QA statistics are written to JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

from c3tta.data.manifest import validate_manifest, write_manifest

DEVICE = "Topcon_TRC-NW400"
OFFICIAL_IMAGE_COUNT = 488
OFFICIAL_PATIENT_COUNT = 244
IMAGE_RE = re.compile(r"^(RET(?P<patient>\d{3}))(?P<eye>OD|OS)\.jpg$", re.IGNORECASE)
CONTOUR_RE = re.compile(
    r"^(?P<image>RET\d{3}(?:OD|OS))_(?P<structure>cup|disc)_exp(?P<expert>[12])\.txt$",
    re.IGNORECASE,
)
PATIENT_CELL_RE = re.compile(r"^#?(?P<number>\d{1,3})$")
SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS = {"x": SPREADSHEET_NS}


@dataclass(frozen=True)
class EyeRecord:
    image_id: str
    patient_id: str
    eye: str
    image_path: Path
    original_diagnosis: int
    glaucoma: int


def _column_index(cell_reference: str) -> int:
    letters = "".join(character for character in cell_reference if character.isalpha())
    if not letters:
        raise ValueError(f"Invalid XLSX cell reference: {cell_reference!r}")
    result = 0
    for character in letters.upper():
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def _read_xlsx_rows(path: Path) -> list[dict[int, str]]:
    """Read cell values from the first XLSX worksheet using only stdlib.

    PAPILA only needs the patient ID and Diagnosis columns.  A small OOXML
    reader avoids making the optional pandas/openpyxl Excel engine a runtime
    requirement for data preparation.
    """

    try:
        archive = ZipFile(path)
    except Exception as exc:  # ZipFile raises several subclasses for bad input.
        raise ValueError(f"Cannot open clinical workbook {path}: {exc}") from exc

    with archive:
        names = set(archive.namelist())
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in names:
            shared_root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("x:si", NS):
                shared_strings.append(
                    "".join(node.text or "" for node in item.findall(".//x:t", NS))
                )

        sheets = sorted(
            name for name in names if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)
        )
        if not sheets:
            raise ValueError(f"Clinical workbook has no worksheet: {path}")
        sheet_root = ElementTree.fromstring(archive.read(sheets[0]))

    rows: list[dict[int, str]] = []
    for row_node in sheet_root.findall(".//x:sheetData/x:row", NS):
        row: dict[int, str] = {}
        for cell in row_node.findall("x:c", NS):
            reference = cell.get("r", "")
            column = _column_index(reference)
            cell_type = cell.get("t", "")
            if cell_type == "inlineStr":
                value = "".join(node.text or "" for node in cell.findall(".//x:t", NS))
            else:
                value_node = cell.find("x:v", NS)
                if value_node is None or value_node.text is None:
                    value = ""
                elif cell_type == "s":
                    try:
                        value = shared_strings[int(value_node.text)]
                    except (IndexError, ValueError) as exc:
                        raise ValueError(
                            f"Invalid shared-string index in {path}, cell {reference}"
                        ) from exc
                else:
                    value = value_node.text
            row[column] = value.strip()
        rows.append(row)
    return rows


def read_clinical_diagnoses(path: str | Path) -> dict[str, int]:
    """Return ``RET### -> {0,1,2}`` from an official PAPILA workbook."""

    path = Path(path)
    rows = _read_xlsx_rows(path)
    diagnosis_columns = {
        column
        for row in rows[:10]
        for column, value in row.items()
        if value.strip().casefold() == "diagnosis"
    }
    if len(diagnosis_columns) != 1:
        raise ValueError(
            f"Expected one Diagnosis column in {path}, found {sorted(diagnosis_columns)}"
        )
    diagnosis_column = diagnosis_columns.pop()

    diagnoses: dict[str, int] = {}
    for row in rows:
        match = PATIENT_CELL_RE.fullmatch(row.get(0, "").strip())
        if match is None:
            continue
        patient_number = int(match.group("number"))
        if not 0 <= patient_number <= 999:
            raise ValueError(f"Patient number outside RET### range in {path}: {patient_number}")
        patient_id = f"RET{patient_number:03d}"
        raw_diagnosis = row.get(diagnosis_column, "")
        try:
            numeric = float(raw_diagnosis)
        except ValueError as exc:
            raise ValueError(
                f"Missing or invalid diagnosis for {patient_id} in {path}: {raw_diagnosis!r}"
            ) from exc
        if not numeric.is_integer() or int(numeric) not in (0, 1, 2):
            raise ValueError(
                f"Diagnosis for {patient_id} in {path} must be 0, 1, or 2; got {numeric}"
            )
        if patient_id in diagnoses:
            raise ValueError(f"Duplicate patient {patient_id} in {path}")
        diagnoses[patient_id] = int(numeric)

    if not diagnoses:
        raise ValueError(f"No PAPILA patient rows found in {path}")
    return diagnoses


def find_dataset_root(path: str | Path) -> Path:
    """Accept either the extracted dataset root or its single wrapper directory."""

    requested = Path(path).expanduser().resolve()
    if not requested.is_dir():
        raise FileNotFoundError(f"PAPILA extraction directory does not exist: {requested}")

    def is_dataset_root(candidate: Path) -> bool:
        return all(
            (candidate / name).is_dir()
            for name in ("ClinicalData", "ExpertsSegmentations", "FundusImages")
        )

    if is_dataset_root(requested):
        return requested
    candidates = sorted(
        {
            directory.parent.resolve()
            for directory in requested.rglob("FundusImages")
            if directory.is_dir() and is_dataset_root(directory.parent)
        }
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one extracted PAPILA dataset below {requested}; "
            f"found {len(candidates)}: {[str(path) for path in candidates]}"
        )
    return candidates[0]


def read_contour(path: str | Path, image_size: tuple[int, int]) -> list[tuple[float, float]]:
    """Read and validate PAPILA ``x y`` vertices for a ``(width, height)`` image."""

    path = Path(path)
    points: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if len(fields) != 2:
                raise ValueError(
                    f"{path}:{line_number}: expected exactly two coordinates, got {fields}"
                )
            try:
                x, y = (float(value) for value in fields)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: non-numeric contour coordinate") from exc
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError(f"{path}:{line_number}: contour coordinate is not finite")
            points.append((x, y))

    if len(points) < 3 or len(set(points)) < 3:
        raise ValueError(f"Contour must contain at least three distinct vertices: {path}")
    width, height = image_size
    outside = [(x, y) for x, y in points if not (0 <= x < width and 0 <= y < height)]
    if outside:
        raise ValueError(
            f"Contour has {len(outside)} vertices outside image size {image_size}: "
            f"{path}; first={outside[0]}"
        )
    return points


def rasterize_contour(path: str | Path, image_size: tuple[int, int]) -> Image.Image:
    """Rasterize a PAPILA polygon without swapping its x/y coordinates."""

    points = read_contour(path, image_size)
    mask = Image.new("L", image_size, color=0)
    ImageDraw.Draw(mask).polygon(points, fill=255)
    if mask.getbbox() is None:
        raise ValueError(f"Rasterized contour is empty: {path}")
    return mask


def _discover_images(dataset_root: Path) -> dict[str, tuple[str, str, Path]]:
    images: dict[str, tuple[str, str, Path]] = {}
    for path in sorted((dataset_root / "FundusImages").iterdir()):
        if not path.is_file():
            continue
        match = IMAGE_RE.fullmatch(path.name)
        if match is None:
            continue
        patient_id = f"RET{match.group('patient')}".upper()
        eye = match.group("eye").upper()
        image_id = f"{patient_id}{eye}"
        if image_id in images:
            raise ValueError(f"Duplicate fundus image ID {image_id}: {images[image_id][2]}, {path}")
        images[image_id] = (patient_id, eye, path)
    if not images:
        raise ValueError(f"No RET###OD/OS.jpg images found under {dataset_root / 'FundusImages'}")
    return images


def _canonical_contours(contour_dir: Path) -> tuple[set[tuple[str, str, int]], list[str]]:
    canonical: set[tuple[str, str, int]] = set()
    ignored: list[str] = []
    for path in sorted(contour_dir.glob("*.txt")):
        match = CONTOUR_RE.fullmatch(path.name)
        if match is None:
            ignored.append(path.name)
            continue
        key = (
            match.group("image").upper(),
            match.group("structure").lower(),
            int(match.group("expert")),
        )
        if key in canonical:
            raise ValueError(f"Duplicate canonical contour key {key} in {contour_dir}")
        canonical.add(key)
    return canonical, ignored


def _validate_inventory(
    dataset_root: Path,
    images: dict[str, tuple[str, str, Path]],
    diagnoses_by_eye: dict[str, dict[str, int]],
    expert: int,
    allow_subset: bool,
) -> list[str]:
    patient_sets = {eye: set(diagnoses) for eye, diagnoses in diagnoses_by_eye.items()}
    if patient_sets["OD"] != patient_sets["OS"]:
        only_od = sorted(patient_sets["OD"] - patient_sets["OS"])
        only_os = sorted(patient_sets["OS"] - patient_sets["OD"])
        raise ValueError(f"OD/OS clinical patient mismatch: only OD={only_od}, only OS={only_os}")

    image_patients = {patient_id for patient_id, _, _ in images.values()}
    clinical_patients = patient_sets["OD"]
    if image_patients != clinical_patients:
        raise ValueError(
            "Clinical/image patient mismatch: "
            f"images-only={sorted(image_patients - clinical_patients)}, "
            f"clinical-only={sorted(clinical_patients - image_patients)}"
        )
    for patient_id in sorted(clinical_patients):
        for eye in ("OD", "OS"):
            image_id = f"{patient_id}{eye}"
            if image_id not in images:
                raise ValueError(f"Missing fundus image for clinical eye {image_id}")

    contour_dir = dataset_root / "ExpertsSegmentations" / "Contours"
    if not contour_dir.is_dir():
        raise FileNotFoundError(f"PAPILA contour directory does not exist: {contour_dir}")
    canonical, ignored = _canonical_contours(contour_dir)
    for image_id in sorted(images):
        for structure in ("disc", "cup"):
            if (image_id, structure, expert) not in canonical:
                raise FileNotFoundError(
                    f"Missing canonical expert-{expert} {structure} contour for {image_id}: "
                    f"{contour_dir / f'{image_id}_{structure}_exp{expert}.txt'}"
                )

    if not allow_subset:
        if len(images) != OFFICIAL_IMAGE_COUNT:
            raise ValueError(
                f"Expected {OFFICIAL_IMAGE_COUNT} official PAPILA images, found {len(images)}. "
                "Use --allow-subset only for an intentional fixture/debug subset."
            )
        if len(clinical_patients) != OFFICIAL_PATIENT_COUNT:
            raise ValueError(
                f"Expected {OFFICIAL_PATIENT_COUNT} official PAPILA patients, "
                f"found {len(clinical_patients)}"
            )
        expected_contour_keys = OFFICIAL_IMAGE_COUNT * 2 * 2
        if len(canonical) != expected_contour_keys:
            raise ValueError(
                f"Expected {expected_contour_keys} canonical contours for both experts, "
                f"found {len(canonical)}"
            )
    return ignored


def _make_records(
    images: dict[str, tuple[str, str, Path]],
    diagnoses_by_eye: dict[str, dict[str, int]],
    suspect_policy: str,
) -> tuple[list[EyeRecord], list[str]]:
    records: list[EyeRecord] = []
    excluded: list[str] = []
    for image_id, (patient_id, eye, path) in sorted(images.items()):
        diagnosis = diagnoses_by_eye[eye][patient_id]
        if diagnosis == 2 and suspect_policy == "exclude":
            excluded.append(image_id)
            continue
        glaucoma = 1 if diagnosis in (1, 2) else 0
        records.append(
            EyeRecord(
                image_id=image_id,
                patient_id=patient_id,
                eye=eye,
                image_path=path,
                original_diagnosis=diagnosis,
                glaucoma=glaucoma,
            )
        )
    if not records:
        raise ValueError("Diagnosis policy excluded every PAPILA eye")
    return records, excluded


def patient_stratified_split(
    records: Iterable[EyeRecord], val_fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    """Split patients, stratifying by whether any included eye is glaucomatous."""

    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")
    patient_labels: dict[str, int] = defaultdict(int)
    for record in records:
        patient_labels[record.patient_id] = max(patient_labels[record.patient_id], record.glaucoma)

    by_label: dict[int, list[str]] = {0: [], 1: []}
    for patient_id, label in sorted(patient_labels.items()):
        by_label[label].append(patient_id)
    if not by_label[0] or not by_label[1]:
        raise ValueError(
            "Patient-level split requires at least one healthy and one glaucoma patient"
        )

    rng = random.Random(seed)
    val_patients: set[str] = set()
    for label in (0, 1):
        patients = by_label[label]
        rng.shuffle(patients)
        if len(patients) < 2:
            raise ValueError(
                f"Patient-level stratification needs at least two class-{label} patients; "
                f"found {len(patients)}"
            )
        count = min(len(patients) - 1, max(1, round(len(patients) * val_fraction)))
        val_patients.update(patients[:count])
    train_patients = set(patient_labels) - val_patients
    if train_patients & val_patients:
        raise AssertionError("Internal error: patient split overlap")
    return train_patients, val_patients


def _place_image(source: Path, destination: Path, mode: str, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {destination}; pass --overwrite")
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(os.path.relpath(source.resolve(), destination.parent.resolve()))
    else:
        raise ValueError(f"Unsupported image mode: {mode}")


def _save_png_atomic(image: Image.Image, path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}; pass --overwrite")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        image.save(temporary, format="PNG")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _labeled_row(record: EyeRecord, split: str, expert: int) -> dict[str, object]:
    return {
        "image": f"images/{record.image_id}.jpg",
        "od_mask": f"masks/expert{expert}/{record.image_id}_od.png",
        "oc_mask": f"masks/expert{expert}/{record.image_id}_oc.png",
        "glaucoma": record.glaucoma,
        "patient_id": record.patient_id,
        "device": DEVICE,
        "split": split,
        "image_id": record.image_id,
        "mask_encoding": "separate_binary",
    }


def _adapt_row(record: EyeRecord) -> dict[str, object]:
    return {
        "image": f"images/{record.image_id}.jpg",
        "patient_id": record.patient_id,
        "device": DEVICE,
        "split": "adapt",
        "image_id": record.image_id,
    }


def _ensure_manifest_outputs_available(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Manifest output(s) already exist; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )


def _validate_outputs(manifests: dict[str, Path], output_root: Path) -> None:
    modes = {"train": "source", "val": "source", "adapt": "target_adapt", "eval": "source"}
    errors: list[str] = []
    for name, path in manifests.items():
        for error in validate_manifest(path, mode=modes[name], check_paths=True, root=output_root):
            errors.append(f"{path}: {error}")
    if errors:
        raise RuntimeError("Generated manifest validation failed:\n" + "\n".join(errors))


def prepare_papila(
    archive_root: str | Path,
    output_root: str | Path,
    manifest_dir: str | Path = "manifests",
    *,
    expert: int = 1,
    suspect_policy: str = "exclude",
    val_fraction: float = 0.2,
    seed: int = 42,
    image_mode: str = "copy",
    overwrite: bool = False,
    allow_subset: bool = False,
) -> dict[str, object]:
    """Convert an extracted PAPILA archive and return reproducibility metadata."""

    if expert not in (1, 2):
        raise ValueError(f"expert must be 1 or 2, got {expert}")
    if suspect_policy not in ("exclude", "positive"):
        raise ValueError("suspect_policy must be 'exclude' or 'positive'")
    if image_mode not in ("copy", "hardlink", "symlink"):
        raise ValueError("image_mode must be copy, hardlink, or symlink")

    dataset_root = find_dataset_root(archive_root)
    output_root = Path(output_root).expanduser().resolve()
    manifest_dir = Path(manifest_dir).expanduser().resolve()
    manifests = {
        name: manifest_dir / f"papila_{name}.csv" for name in ("train", "val", "adapt", "eval")
    }
    metadata_path = manifest_dir / "papila_prepare_metadata.json"
    _ensure_manifest_outputs_available([*manifests.values(), metadata_path], overwrite)

    diagnoses_by_eye = {
        "OD": read_clinical_diagnoses(dataset_root / "ClinicalData" / "patient_data_od.xlsx"),
        "OS": read_clinical_diagnoses(dataset_root / "ClinicalData" / "patient_data_os.xlsx"),
    }
    images = _discover_images(dataset_root)
    ignored_contours = _validate_inventory(
        dataset_root, images, diagnoses_by_eye, expert, allow_subset
    )
    records, excluded_image_ids = _make_records(images, diagnoses_by_eye, suspect_policy)
    train_patients, val_patients = patient_stratified_split(records, val_fraction, seed)

    output_root.mkdir(parents=True, exist_ok=True)
    contour_dir = dataset_root / "ExpertsSegmentations" / "Contours"
    outside_pixels: dict[str, int] = {}
    for record in records:
        with Image.open(record.image_path) as source_image:
            image_size = source_image.size
        disc = rasterize_contour(
            contour_dir / f"{record.image_id}_disc_exp{expert}.txt", image_size
        )
        cup = rasterize_contour(contour_dir / f"{record.image_id}_cup_exp{expert}.txt", image_size)
        disc_array = np.asarray(disc) > 0
        cup_array = np.asarray(cup) > 0
        outside_pixels[record.image_id] = int(np.logical_and(cup_array, ~disc_array).sum())

        _place_image(
            record.image_path,
            output_root / "images" / f"{record.image_id}.jpg",
            image_mode,
            overwrite,
        )
        _save_png_atomic(
            disc,
            output_root / "masks" / f"expert{expert}" / f"{record.image_id}_od.png",
            overwrite,
        )
        _save_png_atomic(
            cup,
            output_root / "masks" / f"expert{expert}" / f"{record.image_id}_oc.png",
            overwrite,
        )

    train_records = [record for record in records if record.patient_id in train_patients]
    val_records = [record for record in records if record.patient_id in val_patients]
    train_label_counts = Counter(record.glaucoma for record in train_records)
    val_label_counts = Counter(record.glaucoma for record in val_records)
    for split_name, label_counts in (
        ("train", train_label_counts),
        ("val", val_label_counts),
    ):
        if not label_counts[0] or not label_counts[1]:
            raise ValueError(
                f"Patient split produced no class-0 or class-1 eyes in {split_name}: "
                f"{dict(label_counts)}"
            )
    train_rows = [_labeled_row(record, "train", expert) for record in train_records]
    val_rows = [_labeled_row(record, "val", expert) for record in val_records]
    adapt_rows = [_adapt_row(record) for record in val_records]
    eval_rows = [_labeled_row(record, "eval", expert) for record in val_records]
    if [row["image_id"] for row in adapt_rows] != [row["image_id"] for row in eval_rows]:
        raise AssertionError("Internal error: adaptation/evaluation order mismatch")

    manifest_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(train_rows, manifests["train"])
    write_manifest(val_rows, manifests["val"])
    write_manifest(adapt_rows, manifests["adapt"])
    write_manifest(eval_rows, manifests["eval"])
    _validate_outputs(manifests, output_root)

    original_counts = Counter(
        diagnoses_by_eye[eye][patient_id] for patient_id, eye, _ in images.values()
    )
    metadata: dict[str, object] = {
        "dataset": "PAPILA",
        "resolved_archive_root": str(dataset_root),
        "prepared_data_root": str(output_root),
        "device": DEVICE,
        "expert": expert,
        "coordinate_order": "x_y",
        "rasterizer": "Pillow.ImageDraw.polygon",
        "cup_was_intersected_with_disc": False,
        "suspect_policy": suspect_policy,
        "diagnosis_mapping": {
            "0": "healthy -> 0",
            "1": "glaucoma -> 1",
            "2": "excluded" if suspect_policy == "exclude" else "suspicious -> 1",
        },
        "split": {
            "unit": "patient",
            "stratification": "any included glaucomatous eye",
            "seed": seed,
            "requested_val_fraction": val_fraction,
        },
        "counts": {
            "original_images": len(images),
            "original_patients": len({item[0] for item in images.values()}),
            "original_diagnosis_0": original_counts[0],
            "original_diagnosis_1": original_counts[1],
            "original_diagnosis_2": original_counts[2],
            "excluded_suspicious_images": len(excluded_image_ids),
            "prepared_images": len(records),
            "train_images": len(train_records),
            "val_images": len(val_records),
            "train_class_0_images": train_label_counts[0],
            "train_class_1_images": train_label_counts[1],
            "val_class_0_images": val_label_counts[0],
            "val_class_1_images": val_label_counts[1],
            "train_patients": len(train_patients),
            "val_patients": len(val_patients),
        },
        "mask_qa": {
            "images_with_cup_pixels_outside_disc": sum(
                count > 0 for count in outside_pixels.values()
            ),
            "total_cup_pixels_outside_disc": sum(outside_pixels.values()),
            "max_cup_pixels_outside_disc": max(outside_pixels.values(), default=0),
            "affected_image_ids": [
                image_id for image_id, count in outside_pixels.items() if count > 0
            ],
        },
        "ignored_noncanonical_contour_files": ignored_contours,
        "manifests": {name: str(path) for name, path in manifests.items()},
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an extracted official PAPILA archive into C3-TTA masks/manifests."
    )
    parser.add_argument(
        "--archive-root",
        required=True,
        help="Extracted PAPILA root or the parent containing its hash-named wrapper directory.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Prepared data root containing copied/linked images and rasterized masks.",
    )
    parser.add_argument("--manifest-dir", default="manifests")
    parser.add_argument("--expert", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--suspect-policy",
        choices=("exclude", "positive"),
        default="exclude",
        help="How diagnosis=2 eyes enter the binary task; the choice is recorded in metadata.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How original JPEGs are placed in the prepared data root.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Disable official 488-image/244-patient inventory checks for fixtures/debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare_papila(
        archive_root=args.archive_root,
        output_root=args.output_root,
        manifest_dir=args.manifest_dir,
        expert=args.expert,
        suspect_policy=args.suspect_policy,
        val_fraction=args.val_fraction,
        seed=args.seed,
        image_mode=args.image_mode,
        overwrite=args.overwrite,
        allow_subset=args.allow_subset,
    )
    print(json.dumps({"status": "ok", **metadata["counts"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
