"""CPU construction of continuous OD/OC targets for DAFL source training."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def normalized_anatomy_field(
    seg_target: torch.Tensor, *, temperature: float
) -> torch.Tensor:
    """Return clipped signed distances divided by the OD vertical diameter.

    ``seg_target`` is expected to hold binary OD then OC masks at the final
    training resolution.  The distance transform is intentionally computed in
    the data worker: it is a fixed target operation and should not consume
    accelerator memory or create a gradient path.
    """

    if seg_target.ndim != 3 or seg_target.shape[0] != 2:
        raise ValueError("seg_target must have shape [2, H, W] for OD and OC")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    mask = (seg_target.detach().cpu().numpy() >= 0.5).astype(np.uint8)
    od_rows = np.flatnonzero(mask[0].any(axis=1))
    if od_rows.size == 0:
        raise ValueError("OD target is empty; cannot normalize anatomy field")
    od_diameter = float(od_rows[-1] - od_rows[0] + 1)
    fields = []
    for channel in range(2):
        inside = np.ascontiguousarray(mask[channel])
        exterior = np.ascontiguousarray(1 - inside)
        distance_inside = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
        distance_outside = cv2.distanceTransform(exterior, cv2.DIST_L2, 3)
        signed_distance = distance_inside - distance_outside
        fields.append(np.tanh(signed_distance / (od_diameter * temperature)))
    return torch.from_numpy(np.stack(fields, axis=0).astype(np.float32, copy=False))
