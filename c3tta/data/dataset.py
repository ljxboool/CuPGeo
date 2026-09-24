from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from .colorist import (
    colorist_config,
    colorist_image,
    rgb_moments_from_path,
)
from .anatomy_field import normalized_anatomy_field
from .manifest import ManifestRow, read_manifest
from .transforms import multiclass_mask_to_images, paired_transform


def _resolve(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def vertical_geometry_target(seg_target: torch.Tensor) -> torch.Tensor:
    """Return normalized top/rim/cup/rim/bottom row allocations.

    Boundaries use pixel-edge coordinates, so the five non-negative lengths
    sum to one and match the inclusive vertical extent used by the scorer.
    This is a fixed source-label target; it is never constructed from target
    domain images during inference.
    """

    if seg_target.ndim == 3:
        seg_target = seg_target.unsqueeze(0)
    if seg_target.ndim != 4 or seg_target.shape[1] != 2:
        raise ValueError("seg_target must have shape [B, 2, H, W]")
    mask = seg_target.detach() >= 0.5
    batch_size, _, height, _ = mask.shape
    result = []
    for index in range(batch_size):
        od_rows = torch.where(mask[index, 0].any(dim=1))[0]
        oc_rows = torch.where(mask[index, 1].any(dim=1))[0]
        if od_rows.numel() == 0 or oc_rows.numel() == 0:
            raise ValueError("vertical geometry targets require non-empty OD and OC masks")
        od_top = od_rows[0].float() / float(height)
        od_bottom = (od_rows[-1].float() + 1.0) / float(height)
        cup_top = torch.maximum(od_top, oc_rows[0].float() / float(height))
        cup_bottom = torch.minimum(od_bottom, (oc_rows[-1].float() + 1.0) / float(height))
        lengths = torch.stack(
            (
                od_top,
                cup_top - od_top,
                cup_bottom - cup_top,
                od_bottom - cup_bottom,
                1.0 - od_bottom,
            )
        ).clamp_min(0.0)
        result.append(lengths / lengths.sum().clamp_min(1e-6))
    return torch.stack(result, dim=0).float()


class FundusDataset(Dataset):
    """CSV-backed fundus dataset.

    Missing masks/labels are represented by None and task-specific losses must
    ignore them. Adaptation datasets are created with ``target_adapt=True`` and
    reject any row containing target labels.
    """

    def __init__(
        self,
        manifest: str | Path,
        root: str | Path = ".",
        image_size: int = 448,
        train: bool = False,
        target_adapt: bool = False,
        augmentation: Mapping[str, Any] | None = None,
        load_targets: bool = True,
        return_anatomy_field: bool = False,
        anatomy_field_temperature: float = 0.05,
        return_vertical_geometry: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest)
        self.root = Path(root)
        self.image_size = image_size
        self.train = train
        self.target_adapt = target_adapt
        self.augmentation = augmentation
        # Strict prediction generation must be able to consume a labeled
        # evaluation manifest without ever opening its masks or emitting
        # targets in the batch.  This is distinct from target adaptation:
        # target adaptation additionally rejects label-bearing rows outright.
        self.load_targets = bool(load_targets)
        self.return_anatomy_field = bool(return_anatomy_field)
        self.anatomy_field_temperature = float(anatomy_field_temperature)
        self.return_vertical_geometry = bool(return_vertical_geometry)
        if self.return_anatomy_field and not self.load_targets:
            raise ValueError("anatomy-field targets require load_targets=true")
        self.rows = read_manifest(self.manifest_path)
        self.colorist = colorist_config(augmentation)
        self.colorist_style_means: np.ndarray | None = None
        self.colorist_style_stds: np.ndarray | None = None
        if self.colorist["enabled"] and (not train or target_adapt):
            raise ValueError(
                "Colorist is source-training-only and cannot be enabled for "
                "validation, evaluation, or target adaptation"
            )
        if target_adapt:
            for row in self.rows:
                if row.glaucoma is not None or row.od_mask or row.oc_mask or row.mask or row.vcdr is not None:
                    raise ValueError(
                        f"Target adaptation row {row.image_id} contains labels; "
                        "use a label-free adaptation manifest."
                    )
        if self.colorist["enabled"]:
            if not self.rows:
                raise ValueError("Colorist requires a non-empty source-training manifest")
            invalid_splits = sorted(
                {row.split for row in self.rows if row.split.lower() != "train"}
            )
            if invalid_splits:
                raise ValueError(
                    "Colorist style bank may contain only split=train rows; found: "
                    + ", ".join(invalid_splits)
                )
            style_moments = []
            for row in self.rows:
                style_path = _resolve(self.root, row.image)
                if style_path is None or not style_path.exists():
                    raise FileNotFoundError(
                        f"Colorist source style image not found for {row.image_id}: "
                        f"{style_path}"
                    )
                style_moments.append(rgb_moments_from_path(style_path))
            self.colorist_style_means = np.stack(
                [mean for mean, _ in style_moments], axis=0
            ).astype(np.float32)
            self.colorist_style_stds = np.stack(
                [std for _, std in style_moments], axis=0
            ).astype(np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = _resolve(self.root, row.image)
        if image_path is None or not image_path.exists():
            raise FileNotFoundError(f"Image not found for {row.image_id}: {image_path}")
        image = Image.open(image_path)
        if self.colorist["enabled"] and torch.rand(()).item() < float(
            self.colorist["probability"]
        ):
            assert self.colorist_style_means is not None
            assert self.colorist_style_stds is not None
            style_index = int(torch.randint(len(self.rows), (1,)).item())
            image = colorist_image(
                image,
                self.colorist_style_means[style_index],
                self.colorist_style_stds[style_index],
                epsilon=float(self.colorist["epsilon"]),
            )
        od = Image.open(_resolve(self.root, row.od_mask)) if self.load_targets and row.od_mask else None
        oc = Image.open(_resolve(self.root, row.oc_mask)) if self.load_targets and row.oc_mask else None
        if self.load_targets and row.mask_encoding.startswith("three_class"):
            mask_path = _resolve(self.root, row.mask or row.od_mask or row.oc_mask)
            if mask_path is None or not mask_path.exists():
                raise ValueError(f"Three-class row has no mask path: {row.image_id}")
            multiclass = Image.open(mask_path)
            od_image, oc_image = multiclass_mask_to_images(multiclass)
            image_t, od_t, oc_t = paired_transform(
                image, od_image, oc_image, self.image_size, self.train, self.augmentation
            )
        else:
            image_t, od_t, oc_t = paired_transform(
                image, od, oc, self.image_size, self.train, self.augmentation
            )
        sample: dict[str, Any] = {
            "image": image_t,
            "image_id": row.image_id,
            "patient_id": row.patient_id,
            "device": row.device,
            "split": row.split,
        }
        if self.load_targets and od_t is not None and oc_t is not None:
            sample["seg_target"] = torch.cat([od_t, oc_t], dim=0)
            if self.return_anatomy_field:
                sample["anatomy_field_target"] = normalized_anatomy_field(
                    sample["seg_target"], temperature=self.anatomy_field_temperature
                )
            if self.return_vertical_geometry:
                sample["vertical_geometry_target"] = vertical_geometry_target(
                    sample["seg_target"]
                ).squeeze(0)
        if self.load_targets and row.glaucoma is not None:
            sample["cls_target"] = torch.tensor([float(row.glaucoma)], dtype=torch.float32)
        if self.load_targets and row.vcdr is not None:
            sample["vcdr_target"] = torch.tensor([row.vcdr], dtype=torch.float32)
        return sample


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "image": torch.stack([x["image"] for x in batch]),
        "image_id": [x["image_id"] for x in batch],
        "patient_id": [x["patient_id"] for x in batch],
        "device": [x["device"] for x in batch],
        "split": [x["split"] for x in batch],
    }
    if all("seg_target" in x for x in batch):
        result["seg_target"] = torch.stack([x["seg_target"] for x in batch])
    if all("cls_target" in x for x in batch):
        result["cls_target"] = torch.stack([x["cls_target"] for x in batch])
    if all("vcdr_target" in x for x in batch):
        result["vcdr_target"] = torch.stack([x["vcdr_target"] for x in batch])
    if all("anatomy_field_target" in x for x in batch):
        result["anatomy_field_target"] = torch.stack(
            [x["anatomy_field_target"] for x in batch]
        )
    if all("vertical_geometry_target" in x for x in batch):
        result["vertical_geometry_target"] = torch.stack(
            [x["vertical_geometry_target"] for x in batch]
        )
    return result


def build_loader(
    manifest: str | Path,
    root: str | Path = ".",
    image_size: int = 448,
    batch_size: int = 2,
    train: bool = False,
    target_adapt: bool = False,
    num_workers: int = 4,
    distributed: bool = False,
    augmentation: Mapping[str, Any] | None = None,
    *,
    load_targets: bool = True,
    return_anatomy_field: bool = False,
    anatomy_field_temperature: float = 0.05,
    return_vertical_geometry: bool = False,
) -> DataLoader:
    dataset = FundusDataset(
        manifest,
        root,
        image_size,
        train,
        target_adapt,
        augmentation,
        load_targets=load_targets,
        return_anatomy_field=return_anatomy_field,
        anatomy_field_temperature=anatomy_field_temperature,
        return_vertical_geometry=return_vertical_geometry,
    )
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(dataset, shuffle=train, drop_last=train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and train),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=train,
        collate_fn=_collate,
    )
