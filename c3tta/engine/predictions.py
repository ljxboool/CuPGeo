"""Versioned, label-free prediction artifacts for deferred evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


PREDICTION_SCHEMA_VERSION = "c3tta.predictions.v1"
_PREDICTION_FIELDS = {
    "seg_logits": torch.float16,
    "cls_logits": torch.float32,
    "soft_vcdr": torch.float32,
    "rim_proxy": torch.float32,
}


def validate_prediction_artifact(artifact: Any) -> dict[str, Any]:
    """Validate and return a ``c3tta.predictions.v1`` artifact."""
    if not isinstance(artifact, dict):
        raise ValueError("Prediction artifact must be a dictionary")
    expected_top_level = {"schema_version", "metadata", "image_ids", "predictions"}
    if set(artifact) != expected_top_level:
        raise ValueError(
            "Prediction artifact keys must be exactly: "
            + ", ".join(sorted(expected_top_level))
        )
    if artifact["schema_version"] != PREDICTION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported prediction schema: {artifact['schema_version']!r}; "
            f"expected {PREDICTION_SCHEMA_VERSION!r}"
        )
    metadata = artifact["metadata"]
    image_ids = artifact["image_ids"]
    predictions = artifact["predictions"]
    if not isinstance(metadata, dict):
        raise ValueError("Prediction artifact metadata must be a dictionary")
    if not isinstance(image_ids, list) or not image_ids:
        raise ValueError("Prediction artifact image_ids must be a non-empty list")
    if any(not isinstance(value, str) or not value for value in image_ids):
        raise ValueError("Prediction artifact image_ids must be non-empty strings")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Prediction artifact contains duplicate image_ids")
    if not isinstance(predictions, dict) or set(predictions) != set(_PREDICTION_FIELDS):
        raise ValueError(
            "Prediction fields must be exactly: "
            + ", ".join(sorted(_PREDICTION_FIELDS))
        )

    count = len(image_ids)
    for name, expected_dtype in _PREDICTION_FIELDS.items():
        value = predictions[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Prediction field {name} must be a tensor")
        if value.device.type != "cpu":
            raise ValueError(f"Prediction field {name} must be stored on CPU")
        if value.dtype != expected_dtype:
            raise ValueError(
                f"Prediction field {name} has dtype {value.dtype}; expected {expected_dtype}"
            )
        if value.ndim == 0 or value.shape[0] != count:
            raise ValueError(f"Prediction field {name} has an invalid sample dimension")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(
                f"Prediction field {name} contains non-finite values; "
                f"{name} contains NaN or infinite values"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"Prediction field {name} contains NaN or infinite values")
    if predictions["seg_logits"].ndim != 4 or predictions["seg_logits"].shape[1] != 2:
        raise ValueError("seg_logits must have shape [N, 2, H, W]")
    for name in ("cls_logits", "soft_vcdr", "rim_proxy"):
        if predictions[name].ndim != 2 or predictions[name].shape[1] != 1:
            raise ValueError(f"{name} must have shape [N, 1]")
    if metadata.get("num_samples") != count:
        raise ValueError("Prediction metadata num_samples does not match image_ids")
    return artifact


def load_prediction_artifact(path: str | Path) -> dict[str, Any]:
    """Load a tensor-only prediction file without permitting arbitrary pickle globals."""
    artifact = torch.load(Path(path), map_location="cpu", weights_only=True)
    return validate_prediction_artifact(artifact)


class PredictionArtifactWriter:
    """Collect model outputs while making label leakage structurally impossible.

    Only explicit model-output attributes and ``image_id`` values are accepted;
    the input batch itself is never stored.
    """

    def __init__(self) -> None:
        self.image_ids: list[str] = []
        self._seen_ids: set[str] = set()
        self._chunks: dict[str, list[torch.Tensor]] = {
            name: [] for name in _PREDICTION_FIELDS
        }

    def append(self, image_ids: Sequence[str], output: Any) -> None:
        ids = [str(image_id) for image_id in image_ids]
        if not ids or any(not image_id for image_id in ids):
            raise ValueError("Prediction batches require non-empty image_id values")
        duplicates = self._seen_ids.intersection(ids)
        if len(set(ids)) != len(ids) or duplicates:
            duplicate = sorted(duplicates or {x for x in ids if ids.count(x) > 1})[0]
            raise ValueError(f"Duplicate prediction image_id: {duplicate}")

        tensors: dict[str, torch.Tensor] = {}
        for name, dtype in _PREDICTION_FIELDS.items():
            value = getattr(output, name, None)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Model output {name} must be a tensor")
            if value.ndim == 0 or value.shape[0] != len(ids):
                raise ValueError(
                    f"Model output {name} has batch {tuple(value.shape)} for {len(ids)} image_ids"
                )
            tensors[name] = value.detach().to(device="cpu", dtype=dtype).contiguous()

        self.image_ids.extend(ids)
        self._seen_ids.update(ids)
        for name, value in tensors.items():
            self._chunks[name].append(value)

    def build(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        if not self.image_ids:
            raise ValueError("Cannot build an empty prediction artifact")
        predictions = {
            name: torch.cat(chunks, dim=0) for name, chunks in self._chunks.items()
        }
        count = len(self.image_ids)
        if any(value.shape[0] != count for value in predictions.values()):
            raise RuntimeError("Prediction artifact fields have inconsistent sample counts")
        artifact_metadata = dict(metadata)
        artifact_metadata["num_samples"] = count
        return validate_prediction_artifact({
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "metadata": artifact_metadata,
            "image_ids": list(self.image_ids),
            "predictions": predictions,
        })

    def save(self, path: str | Path, metadata: Mapping[str, Any]) -> Path:
        """Atomically save the artifact so scorers never observe a partial file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            torch.save(self.build(metadata), temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path
