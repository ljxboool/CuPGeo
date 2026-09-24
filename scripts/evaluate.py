#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from c3tta.data.dataset import build_loader
from c3tta.engine.common import autocast_context, load_trusted_checkpoint, sha256_file
from c3tta.engine.inference import predict_with_tta
from c3tta.engine.predictions import PredictionArtifactWriter
from c3tta.models.config import load_config
from c3tta.models.factory import build_model, validate_checkpoint_architecture


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default=None,
        help=(
            "Override checkpoint/config inference precision for the current hardware. "
            "Useful when evaluating a BF16-trained checkpoint on FP16-only GPUs such as V100."
        ),
    )
    parser.add_argument(
        "--frozen-backbone-from-config",
        action="store_true",
        help=(
            "Initialize the frozen backbone from model.checkpoint and allow the runtime "
            "checkpoint to omit frozen backbone tensors. All non-backbone and LoRA tensors "
            "remain mandatory."
        ),
    )
    parser.add_argument(
        "--tta-image-size",
        action="append",
        type=int,
        default=None,
        help=(
            "Optional inference view size; repeat to enable label-free multi-scale "
            "logit ensembling. Defaults to the configured input size."
        ),
    )
    parser.add_argument(
        "--tta-horizontal-flip",
        action="store_true",
        help="Include a horizontally flipped label-free inference view in every scale.",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help=(
            "Optional label-free prediction artifact. When set, inference does "
            "not import or calculate metrics; use score_predictions.py later."
        ),
    )
    parser.add_argument(
        "--artifact-method",
        default="source-only",
        help="Method label stored in a --predictions artifact.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    checkpoint = load_trusted_checkpoint(args.checkpoint, map_location="cpu")
    saved_config = checkpoint.get("config", {})
    if saved_config.get("model"):
        cfg.model.update(saved_config["model"])
    if saved_config.get("train", {}).get("precision"):
        cfg.train["precision"] = saved_config["train"]["precision"]
    if args.precision is not None:
        cfg.train["precision"] = args.precision
    validate_checkpoint_architecture(checkpoint, cfg)
    if args.data_root:
        cfg.data["root"] = args.data_root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.frozen_backbone_from_config and not cfg.model.get("freeze_backbone", False):
        raise ValueError("--frozen-backbone-from-config requires model.freeze_backbone=true")
    model = build_model(
        cfg,
        initialize_backbone=args.frozen_backbone_from_config,
    ).to(device)
    if args.frozen_backbone_from_config:
        incompatible = model.load_state_dict(checkpoint["model"], strict=False)
        invalid_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith("backbone.model.") or ".lora_" in key
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Partial checkpoint is missing required tensors or contains unexpected tensors: "
                f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
            )
    else:
        model.load_state_dict(checkpoint["model"], strict=True)
    workers = args.num_workers if args.num_workers is not None else cfg.train.get("num_workers", 4)
    # Artifact generation is deliberately image-only: the manifest may carry
    # hidden evaluation labels for later offline scoring, but inference must
    # neither open them nor place them in a batch.
    loader = build_loader(
        args.csv,
        cfg.data.get("root", "."),
        cfg.model.get("image_size", 448),
        args.batch_size or cfg.train.get("batch_size", 2),
        False,
        False,
        workers,
        False,
        load_targets=not bool(args.predictions),
    )
    if args.predictions:
        writer = PredictionArtifactWriter()
        model.eval()
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                with autocast_context(device, cfg.train.get("precision", "fp16")):
                    prediction = predict_with_tta(
                        model,
                        images,
                        image_sizes=args.tta_image_size,
                        horizontal_flip=args.tta_horizontal_flip,
                    )
                writer.append(batch["image_id"], prediction)
        checkpoint_metrics = checkpoint.get("metrics", {})
        if not isinstance(checkpoint_metrics, dict):
            checkpoint_metrics = {}
        checkpoint_train = saved_config.get("train", {})
        if not isinstance(checkpoint_train, dict):
            checkpoint_train = {}
        source_seed = checkpoint_metrics.get("seed", checkpoint_train.get("seed"))
        prediction_path = writer.save(
            args.predictions,
            {
                "method": args.artifact_method,
                "base_method": "source-only",
                "optimization_applied": False,
                "inference_target_labels_loaded": False,
                "seed": int(source_seed) if source_seed is not None else None,
                "source_seed": int(source_seed) if source_seed is not None else None,
                "checkpoint_sha256": sha256_file(args.checkpoint),
                "checkpoint_architecture": checkpoint.get("model_architecture"),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "checkpoint_selection_metric": checkpoint.get("selection_metric"),
                "checkpoint_selection_score": checkpoint.get("best_selection_score"),
                "checkpoint_training_precision": checkpoint_train.get("precision"),
                "config_filename": Path(args.config).name,
                "config_sha256": sha256_file(args.config),
                "inference_manifest_filename": Path(args.csv).name,
                "inference_manifest_sha256": sha256_file(args.csv),
                "inference_precision": cfg.train.get("precision", "fp16"),
                "frozen_backbone_from_config": bool(args.frozen_backbone_from_config),
                "artifact_storage_dtypes": {
                    "seg_logits": "float16",
                    "cls_logits": "float32",
                    "soft_vcdr": "float32",
                    "rim_proxy": "float32",
                },
                "inference_tta_image_sizes": (
                    [int(value) for value in args.tta_image_size]
                    if args.tta_image_size is not None
                    else [int(cfg.model.get("image_size", 448))]
                ),
                "inference_tta_horizontal_flip": bool(args.tta_horizontal_flip),
            },
        )
        result = {
            "method": args.artifact_method,
            "prediction_artifact": str(prediction_path),
            "num_predictions": len(writer.image_ids),
            "optimization_applied": False,
        }
    else:
        # Import lazily so source inference can run in the CUDA environment
        # even when metric-only dependencies live in a separate scorer env.
        from c3tta.engine.evaluation import evaluate_loader

        result = evaluate_loader(model, loader, device, cfg.train.get("precision", "fp16"))
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
