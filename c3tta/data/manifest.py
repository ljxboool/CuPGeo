"""Manifest IO and safety checks.

The adaptation manifest intentionally has a smaller schema than the evaluation
manifest. This makes accidental target-label access fail early.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import csv


# The mask columns are checked per row because REFUGE-style exports commonly
# store either two binary masks or one 0/1/2 label image.  Requiring both mask
# columns at the CSV-header level would incorrectly reject the latter format.
SOURCE_REQUIRED = {"image", "glaucoma", "patient_id", "device", "split", "image_id"}
TARGET_ADAPT_REQUIRED = {"image", "patient_id", "device", "split", "image_id"}
TARGET_FORBIDDEN = {
    "glaucoma",
    "od_mask",
    "oc_mask",
    "mask",
    "mask_path",
    "seg_mask",
    "fovea_x",
    "fovea_y",
    "vcdr",
}


@dataclass(frozen=True)
class ManifestRow:
    image: str
    patient_id: str
    device: str
    split: str
    image_id: str
    od_mask: str | None = None
    oc_mask: str | None = None
    mask: str | None = None
    glaucoma: int | None = None
    vcdr: float | None = None
    mask_encoding: str = "separate_binary"
    source_domain: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, str]) -> "ManifestRow":
        def optional(key: str) -> str | None:
            value = raw.get(key, "")
            return value if value else None

        label = optional("glaucoma")
        vcdr_raw = optional("vcdr")
        return cls(
            image=raw["image"],
            patient_id=raw["patient_id"],
            device=raw["device"],
            split=raw["split"],
            image_id=raw["image_id"],
            od_mask=optional("od_mask"),
            oc_mask=optional("oc_mask"),
            mask=optional("mask") or optional("mask_path") or optional("seg_mask"),
            glaucoma=None if label is None else int(float(label)),
            vcdr=None if vcdr_raw is None else float(vcdr_raw),
            mask_encoding=raw.get("mask_encoding") or "separate_binary",
            source_domain=optional("source_domain"),
        )


def read_manifest(path: str | Path) -> list[ManifestRow]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        return [ManifestRow.from_dict(row) for row in reader]


def validate_manifest(
    path: str | Path,
    mode: str = "source",
    check_paths: bool = False,
    root: str | Path = ".",
    require_unique_ids: bool = True,
) -> list[str]:
    """Return validation errors; raise nothing so CLI can print all failures."""
    path = Path(path)
    root = Path(root)
    errors: list[str] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = SOURCE_REQUIRED if mode == "source" else TARGET_ADAPT_REQUIRED
        missing = sorted(required - fields)
        if missing:
            errors.append(f"missing required columns: {', '.join(missing)}")
        if mode == "target_adapt":
            present_forbidden = sorted(TARGET_FORBIDDEN & fields)
            if present_forbidden:
                errors.append(
                    "target adaptation manifest contains forbidden label columns: "
                    + ", ".join(present_forbidden)
                )

        seen: set[str] = set()
        patients_by_split: dict[str, set[str]] = {}
        for line_no, raw in enumerate(reader, start=2):
            try:
                row = ManifestRow.from_dict(raw)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"line {line_no}: invalid row: {exc}")
                continue
            if mode == "source":
                encoding = row.mask_encoding.lower()
                if encoding.startswith("three_class"):
                    if not (row.mask or row.od_mask or row.oc_mask):
                        errors.append(
                            f"line {line_no}: three_class rows require mask/mask_path "
                            "(or an od_mask-compatible path)"
                        )
                elif not (row.od_mask and row.oc_mask):
                    errors.append(
                        f"line {line_no}: separate_binary rows require both od_mask and oc_mask"
                    )
            if row.glaucoma not in (None, 0, 1):
                errors.append(f"line {line_no}: glaucoma must be 0/1")
            if require_unique_ids and row.image_id in seen:
                errors.append(f"line {line_no}: duplicate image_id={row.image_id}")
            seen.add(row.image_id)
            patients_by_split.setdefault(row.split, set()).add(row.patient_id)
            if check_paths:
                for name in ("image", "od_mask", "oc_mask", "mask"):
                    value = getattr(row, name)
                    if value:
                        resolved = Path(value) if Path(value).is_absolute() else root / value
                        if not resolved.exists():
                            errors.append(f"line {line_no}: {name} does not exist: {resolved}")

        split_names = sorted(patients_by_split)
        for i, left in enumerate(split_names):
            for right in split_names[i + 1 :]:
                overlap = patients_by_split[left] & patients_by_split[right]
                if overlap:
                    errors.append(
                        f"patient leakage between {left} and {right}: {sorted(overlap)[:5]}"
                    )
    return errors


def write_manifest(rows: Iterable[dict[str, object]], path: str | Path) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty manifest")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
