#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from c3tta.engine.common import load_trusted_checkpoint, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create an inference-only checkpoint that omits an unchanged frozen timm backbone."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_path = Path(args.checkpoint)
    output_path = Path(args.output)
    checkpoint = load_trusted_checkpoint(source_path, map_location="cpu")
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})
    if not model_config.get("freeze_backbone", False):
        raise ValueError("Delta export requires model.freeze_backbone=true")

    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict):
        raise ValueError("Checkpoint does not contain a model state dictionary")
    delta_state = {
        key: value
        for key, value in model_state.items()
        if not key.startswith("backbone.model.") or ".lora_" in key
    }

    payload = {
        key: checkpoint[key]
        for key in (
            "model_architecture",
            "config",
            "epoch",
            "best_val_loss",
            "best_selection_score",
            "selection_metric",
            "metrics",
        )
        if key in checkpoint
    }
    payload.update(
        {
            "model": delta_state,
            "checkpoint_format": "c3tta.frozen_backbone_delta.v1",
            "source_checkpoint_sha256": sha256_file(source_path),
            "num_full_model_tensors": len(model_state),
            "num_delta_tensors": len(delta_state),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(
        {
            "output": str(output_path),
            "num_full_model_tensors": len(model_state),
            "num_delta_tensors": len(delta_state),
            "output_bytes": output_path.stat().st_size,
        }
    )


if __name__ == "__main__":
    main()
