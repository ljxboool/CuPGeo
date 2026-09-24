from __future__ import annotations

from pathlib import Path
import csv
import math
import random

import numpy as np
from PIL import Image, ImageDraw


def make_synthetic_manifest(out_dir: str | Path, count: int = 8, image_size: int = 128) -> tuple[Path, Path]:
    """Create tiny deterministic data for CI/smoke tests, not paper results."""
    out = Path(out_dir)
    image_dir, mask_dir = out / "images", out / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(count):
        rng = random.Random(idx)
        cx, cy = image_size // 2 + rng.randint(-8, 8), image_size // 2 + rng.randint(-5, 5)
        rx, ry = image_size // 3, image_size // 4
        crx, cry = max(6, rx // 2), max(6, ry // 2)
        image = Image.new("RGB", (image_size, image_size), (30, 35, 45))
        draw = ImageDraw.Draw(image)
        draw.ellipse((cx-rx, cy-ry, cx+rx, cy+ry), fill=(180, 105, 80))
        draw.ellipse((cx-crx, cy-cry, cx+crx, cy+cry), fill=(230, 195, 170))
        od = Image.new("L", (image_size, image_size), 0)
        oc = Image.new("L", (image_size, image_size), 0)
        ImageDraw.Draw(od).ellipse((cx-rx, cy-ry, cx+rx, cy+ry), fill=255)
        ImageDraw.Draw(oc).ellipse((cx-crx, cy-cry, cx+crx, cy+cry), fill=255)
        image_id = f"synthetic_{idx:04d}"
        image.save(image_dir / f"{image_id}.png")
        od.save(mask_dir / f"{image_id}_od.png")
        oc.save(mask_dir / f"{image_id}_oc.png")
        rows.append({"image": str(Path("images") / f"{image_id}.png"), "od_mask": str(Path("masks") / f"{image_id}_od.png"), "oc_mask": str(Path("masks") / f"{image_id}_oc.png"), "glaucoma": int(idx % 2), "patient_id": f"p{idx//2}", "device": "synthetic", "split": "train" if idx < count * 3 // 4 else "val", "image_id": image_id, "mask_encoding": "separate_binary"})
    manifest = out / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    target = out / "target_adapt.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        fields = ["image", "patient_id", "device", "split", "image_id"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    return manifest, target
