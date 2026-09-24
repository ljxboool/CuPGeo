#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from c3tta.data.dataset import build_loader, vertical_geometry_target as make_vertical_target
from c3tta.data.colorist import colorist_config
from c3tta.engine.common import autocast_context, load_trusted_checkpoint, make_grad_scaler, seed_everything, unwrap
from c3tta.metrics import classification_metrics
from c3tta.models.config import load_config
from c3tta.models.factory import (
    build_model,
    model_architecture_for_config,
    segmentation_hierarchy_for_config,
    validate_checkpoint_architecture,
    validate_init_checkpoint_architecture,
)
from c3tta.models.multitask import INDEPENDENT_SEGMENTATION
from c3tta.losses.counterfactual import (
    camera_style_counterfactual,
    paired_counterfactual_config,
    paired_counterfactual_consistency_loss,
)
from c3tta.losses.scale_equivariance import (
    paired_scale_equivariance_config,
    paired_scale_equivariance_consistency_loss,
    scale_canvas_batch,
)
from c3tta.losses.anatomy_field import (
    field_mask_consistency_loss,
    paired_scale_anatomy_field_loss,
    scale_normalized_anatomy_field_config,
)
from c3tta.losses.hierarchy import soft_topology_config, thresholded_topology_loss
from c3tta.losses.multitask import source_loss
from c3tta.losses.multitask import vertical_geometry_loss


def lr_multiplier(train_config: dict, epoch: int, total_epochs: int) -> float:
    """Return the deterministic per-epoch source-training LR scale.

    The default deliberately remains a constant LR.  Cosine decay is exposed
    as an opt-in source-only ablation because this small frozen-backbone setup
    can otherwise keep fitting after its validation optimum.
    """
    schedule = str(train_config.get("lr_schedule", "none")).lower()
    if schedule in {"", "none", "constant"}:
        return 1.0
    if schedule != "cosine":
        raise ValueError(f"Unsupported lr_schedule: {schedule!r}")
    warmup_epochs = int(train_config.get("lr_warmup_epochs", 0))
    min_scale = float(train_config.get("lr_min_scale", 0.0))
    if warmup_epochs < 0:
        raise ValueError("lr_warmup_epochs must be non-negative")
    if not 0.0 <= min_scale <= 1.0:
        raise ValueError("lr_min_scale must be in [0, 1]")
    if warmup_epochs and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    decay_epochs = max(total_epochs - warmup_epochs - 1, 1)
    progress = min(max(epoch - warmup_epochs, 0) / decay_epochs, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_scale + (1.0 - min_scale) * cosine


def auxiliary_loss_warmup(epoch: int, warmup_epochs: int) -> float:
    """Linearly turn on a geometry auxiliary objective after mask stabilization."""

    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")
    if warmup_epochs == 0:
        return 1.0
    return min(max(float(epoch) / float(warmup_epochs), 0.0), 1.0)


def delayed_auxiliary_loss_ramp(
    epoch: int, *, start_epoch: int, ramp_epochs: int
) -> float:
    """Keep an auxiliary loss off, then linearly ramp it to full weight."""

    if start_epoch < 0 or ramp_epochs < 0:
        raise ValueError("start_epoch and ramp_epochs must be non-negative")
    if epoch < start_epoch:
        return 0.0
    return auxiliary_loss_warmup(epoch - start_epoch, ramp_epochs)


def source_loss_kwargs(train_config: dict) -> dict[str, float]:
    """Read optional, source-label-only multitask loss weights from config."""
    configured = train_config.get("loss", {})
    if not isinstance(configured, dict):
        raise ValueError("train.loss must be a mapping when provided")
    defaults = {
        "seg_bce_weight": 1.0,
        "seg_dice_weight": 1.0,
        "cls_weight": 1.0,
        "nest_weight": 0.0,
        "boundary_bce_weight": 0.0,
        "boundary_dice_weight": 0.0,
        "vcdr_weight": 0.0,
    }
    result = {name: float(configured.get(name, default)) for name, default in defaults.items()}
    if any(weight < 0.0 for weight in result.values()):
        raise ValueError("source loss weights must be non-negative")
    return result


def checkpoint_selection(train_config: dict) -> tuple[str, bool]:
    """Return (metric_name, higher_is_better) for source validation only."""
    metric = str(train_config.get("selection_metric", "val_loss"))
    options = {
        "val_loss": False,
        "val_seg_dice": True,
        "val_auroc": True,
        "val_auprc": True,
    }
    if metric not in options:
        raise ValueError(
            f"Unsupported selection_metric {metric!r}; choose one of {sorted(options)}"
        )
    return metric, options[metric]


def validation_classification_metrics(
    logits: list[torch.Tensor], targets: list[torch.Tensor]
) -> dict[str, float]:
    """Compute source-validation classification metrics over all samples.

    Ranking metrics are non-additive, so callers must collect the complete
    source-validation set rather than average per-batch AUROC/AUPRC values.
    Logits are promoted before sigmoid to keep checkpoint selection independent
    of FP16 sigmoid quantization.
    """
    if not logits:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return classification_metrics(
        torch.cat(logits, dim=0).float(),
        torch.cat(targets, dim=0).float(),
    )


def _all_reduce_sum_count(
    total: float, count: int, device: torch.device, distributed: bool
) -> tuple[float, int]:
    """Reduce a scalar sum and its sample count, avoiding rank-local metrics."""
    values = torch.tensor([total, float(count)], dtype=torch.float64, device=device)
    if distributed:
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    return float(values[0].item()), int(values[1].item())


def _rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": __import__("numpy").random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, object]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    __import__("numpy").random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def init_distributed() -> tuple[int, int, int, bool]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world > 1
    if distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        # NCCL must see the rank's CUDA device before its process group (and
        # the first barrier) is initialized.  Initializing first can leave
        # device selection ambiguous on multi-GPU hosts.
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if backend == "nccl":
            try:
                torch.distributed.init_process_group(backend, device_id=local_rank)
            except TypeError:
                # ``device_id`` was added after the oldest supported PyTorch
                # releases; the CUDA device above is still bound for them.
                torch.distributed.init_process_group(backend)
        else:
            torch.distributed.init_process_group(backend)
    else:
        rank = local_rank = 0
    return rank, local_rank, world, distributed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--backbone-checkpoint", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume", default=None, help="resume from a last.pt/best.pt checkpoint")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help=(
            "initialize a compatible zero-initialized architecture expansion from a trusted "
            "source checkpoint without restoring optimizer/RNG state"
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.backbone:
        cfg.model["backbone"] = args.backbone
    if args.backbone_checkpoint:
        cfg.model["checkpoint"] = args.backbone_checkpoint
    if args.image_size:
        cfg.model["image_size"] = args.image_size
    if args.train_manifest:
        cfg.data["train_manifest"] = args.train_manifest
    if args.val_manifest:
        cfg.data["val_manifest"] = args.val_manifest
    if args.data_root:
        cfg.data["root"] = args.data_root
    for key, value in (("batch_size", args.batch_size), ("grad_accum_steps", args.grad_accum_steps), ("epochs", args.epochs), ("num_workers", args.num_workers), ("precision", args.precision)):
        if value is not None:
            cfg.train[key] = value
    if args.gradient_checkpointing:
        cfg.train["gradient_checkpointing"] = True
    if args.compile:
        cfg.hardware["compile"] = True
    if args.smoke:
        cfg.train["epochs"] = min(cfg.train.get("epochs", 2), 2)
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if cfg.train.get("precision") == "bf16" and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            raise RuntimeError("BF16 profile requires Ampere or newer GPU; use the V100 FP16 profile")

    colorist = colorist_config(cfg.train.get("augmentation"))
    counterfactual = paired_counterfactual_config(cfg.train)
    scale_equivariance = paired_scale_equivariance_config(cfg.train)
    anatomy_field = scale_normalized_anatomy_field_config(cfg.train)
    soft_topology = soft_topology_config(cfg.train)
    vra_config = cfg.train.get("vertical_rim_allocation", {}) or {}
    if not isinstance(vra_config, dict):
        raise ValueError("train.vertical_rim_allocation must be a mapping")
    vra_enabled = bool(vra_config.get("enabled", False))
    vra_weight = float(vra_config.get("weight", 0.0))
    vra_log_margin_weight = float(vra_config.get("log_margin_weight", 0.5))
    if vra_enabled and not bool(cfg.model.get("vertical_geometry_head", False)):
        raise ValueError("enabled vertical rim allocation requires model.vertical_geometry_head=true")
    if vra_enabled and vra_weight <= 0.0:
        raise ValueError("enabled vertical rim allocation requires a positive weight")
    if vra_weight < 0.0 or vra_log_margin_weight < 0.0:
        raise ValueError("vertical rim allocation weights must be non-negative")
    loss_kwargs = source_loss_kwargs(cfg.train)
    if colorist["enabled"] and counterfactual["enabled"]:
        raise ValueError(
            "Colorist is a single-view supervised baseline and cannot be combined "
            "with train.paired_counterfactual"
        )
    if scale_equivariance["enabled"]:
        if colorist["enabled"] or counterfactual["enabled"]:
            raise ValueError(
                "paired scale equivariance is a distinct source-only pilot and "
                "cannot be combined with Colorist or paired camera counterfactuals"
            )
        augmentation = cfg.train.get("augmentation", {}) or {}
        if float(augmentation.get("scale_canvas_prob", 0.0)) > 0.0:
            raise ValueError(
                "paired scale equivariance requires an unscaled clean anchor; "
                "disable single-view scale_canvas augmentation"
            )
    if anatomy_field["enabled"]:
        if not scale_equivariance["enabled"]:
            raise ValueError(
                "scale-normalized anatomy fields require paired_scale_equivariance"
            )
        if not bool(cfg.model.get("anatomy_field_head", False)):
            raise ValueError(
                "enabled scale-normalized anatomy fields require model.anatomy_field_head=true"
            )
    if soft_topology["enabled"]:
        if segmentation_hierarchy_for_config(cfg) != INDEPENDENT_SEGMENTATION:
            raise ValueError(
                "train.soft_topology requires independent segmentation logits; "
                "disable hard model hierarchy"
            )
        if loss_kwargs["nest_weight"] > 0.0:
            raise ValueError(
                "train.soft_topology and legacy train.loss.nest_weight are mutually "
                "exclusive"
            )

    rank, local_rank, world, distributed = init_distributed()
    base_seed = args.seed if args.seed is not None else cfg.train.get("seed", 0)
    seed_everything(int(base_seed) + rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    out_dir = Path(args.output)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        torch.distributed.barrier()

    model = build_model(cfg)
    if args.init_checkpoint:
        init_state = load_trusted_checkpoint(args.init_checkpoint, map_location="cpu")
        # Accept an exact architecture or one of the two audited zero-init
        # expansions that adds global_classifier.* or the shared Pyramid-FPN
        # decoder.tta_adapter.* namespace.  Matching tensor shapes alone are
        # insufficient because hierarchy modes assign different semantics to
        # the same two segmentation channels.
        expansion_prefixes = validate_init_checkpoint_architecture(init_state, cfg)
        incompatible = model.load_state_dict(init_state["model"], strict=False)
        expected_missing = {
            name
            for name in model.state_dict()
            if name.startswith(expansion_prefixes)
        }
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Init checkpoint is not compatible with the requested model expansion: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        if rank == 0:
            print(
                json.dumps(
                    {
                        "init_checkpoint": args.init_checkpoint,
                        "init_architecture": init_state.get("model_architecture"),
                        "initialized_missing_keys": incompatible.missing_keys,
                    }
                ),
                flush=True,
            )
    model = model.to(device)
    base = model
    trainable_scope = str(cfg.train.get("trainable_scope", "source"))
    if trainable_scope == "global_classifier":
        base.freeze_all_except_global_classifier()
    elif trainable_scope != "source":
        raise ValueError("trainable_scope must be 'source' or 'global_classifier'")
    elif cfg.model.get("freeze_backbone", True):
        base.freeze_backbone()
    loss_config = cfg.train.get("loss", {}) or {}
    segmentation_only = float(loss_config.get("cls_weight", 1.0)) == 0.0
    if segmentation_only:
        # Segmentation-only public benchmarks deliberately retain the
        # classification branch for architecture compatibility, but provide no
        # classification loss.  Freeze every classifier-only parameter before
        # DDP/optimizer construction so the distributed graph contains only
        # parameters with a supervised loss path.
        frozen_heads = []
        for name in ("classifier", "global_classifier"):
            module = getattr(base, name, None)
            if module is None:
                continue
            parameters = list(module.parameters())
            if parameters:
                for parameter in parameters:
                    parameter.requires_grad_(False)
                frozen_heads.append(name)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "segmentation_only": True,
                        "frozen_unsupervised_heads": frozen_heads,
                    }
                ),
                flush=True,
            )
    if cfg.train.get("gradient_checkpointing", False):
        base.enable_gradient_checkpointing()
    if cfg.hardware.get("compile", False) and hasattr(torch, "compile") and not args.smoke:
        model = torch.compile(model)
    if distributed:
        ddp_kwargs = {"find_unused_parameters": False}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [local_rank]
        model = DDP(model, **ddp_kwargs)
    base = unwrap(model)
    if args.smoke and cfg.model.get("backbone") != "tiny":
        raise ValueError("--smoke requires --backbone tiny or a tiny backbone profile")

    resume_state = None
    start_epoch = 0
    if args.resume:
        resume_state = load_trusted_checkpoint(args.resume, map_location="cpu")
        validate_checkpoint_architecture(resume_state, cfg)
        base.load_state_dict(resume_state["model"], strict=True)
        start_epoch = int(resume_state.get("epoch", -1)) + 1
        if rank == 0:
            print(json.dumps({"resume": args.resume, "start_epoch": start_epoch}), flush=True)

    train_loader = build_loader(
        cfg.data["train_manifest"],
        cfg.data.get("root", "."),
        cfg.model.get("image_size", 448),
        cfg.train.get("batch_size", 2),
        True,
        False,
        cfg.train.get("num_workers", 4),
        distributed,
        cfg.train.get("augmentation"),
        return_anatomy_field=anatomy_field["enabled"],
        anatomy_field_temperature=anatomy_field["temperature"],
        return_vertical_geometry=vra_enabled,
    )
    if rank == 0 and colorist["enabled"]:
        print(
            json.dumps(
                {
                    "colorist": colorist,
                    "style_pool_size": len(train_loader.dataset),
                    "implementation": "independent_from_arxiv_2608.18915",
                }
            ),
            flush=True,
        )
    val_loader = build_loader(cfg.data["val_manifest"], cfg.data.get("root", "."), cfg.model.get("image_size", 448), cfg.train.get("batch_size", 2), False, False, cfg.train.get("num_workers", 4), False)
    # The legacy decoder splits its adapter LR from its fusion LR; the new
    # pyramid decoder intentionally puts every decoder parameter in the dense
    # decoder group.  This avoids silently skipping layers when architectures
    # evolve.
    adapter_parameters = [
        parameter
        for module in getattr(base.decoder, "adapters", ())
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    adapter_ids = {id(parameter) for parameter in adapter_parameters}
    lora_parameters = [
        parameter
        for parameter in getattr(base.backbone, "trainable_lora_parameters", lambda: [])()
        if parameter.requires_grad
    ]
    dense_decoder_parameters = [
        parameter
        for parameter in base.decoder.parameters()
        if parameter.requires_grad and id(parameter) not in adapter_ids
    ]
    groups = [
        {"params": dense_decoder_parameters, "lr": cfg.train.get("lr_decoder", 1e-3)},
        {"params": adapter_parameters, "lr": cfg.train.get("lr_adapter", 3e-4)},
        {"params": [p for p in base.classifier.parameters() if p.requires_grad], "lr": cfg.train.get("lr_classifier", 1e-3)},
    ]
    global_classifier = getattr(base, "global_classifier", None)
    global_classifier_parameters = (
        [parameter for parameter in global_classifier.parameters() if parameter.requires_grad]
        if global_classifier is not None
        else []
    )
    groups.append(
        {
            "params": global_classifier_parameters,
            "lr": cfg.train.get("lr_global_classifier", cfg.train.get("lr_classifier", 1e-3)),
        }
    )
    vertical_geometry_parameters = (
        [base.vertical_geometry_alpha]
        if getattr(base, "vertical_geometry_alpha", None) is not None
        and base.vertical_geometry_alpha.requires_grad
        else []
    )
    groups.append(
        {
            "params": vertical_geometry_parameters,
            "lr": cfg.train.get("lr_vertical_geometry", cfg.train.get("lr_decoder", 1e-3)),
        }
    )
    if trainable_scope == "source" and cfg.model.get("freeze_backbone", True):
        groups.append({"params": lora_parameters, "lr": cfg.train.get("lr_lora", 1e-4)})
    elif trainable_scope == "source":
        groups.append({"params": [p for p in base.backbone.parameters() if p.requires_grad], "lr": cfg.train.get("lr_backbone", 1e-5)})
    optimizer = AdamW([group for group in groups if group["params"]], weight_decay=cfg.train.get("weight_decay", 1e-4))
    grouped_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    ungrouped = [
        name for name, parameter in base.named_parameters()
        if parameter.requires_grad and id(parameter) not in grouped_parameter_ids
    ]
    if ungrouped:
        raise RuntimeError(f"Trainable parameters missing from optimizer: {ungrouped}")
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if rank == 0:
        style_augmentation = getattr(base.decoder, "style_augmentation", None)
        style_record = (
            style_augmentation.describe()
            if hasattr(style_augmentation, "describe")
            else {"method": "none", "evaluation_is_identity": True}
        )
        print(
            json.dumps(
                {
                    "model_architecture": model_architecture_for_config(cfg),
                    "decoder_type": cfg.model.get("decoder_type", "adapter_fusion"),
                    "trainable_scope": trainable_scope,
                    "parameters": base.trainable_parameter_summary(),
                    "optimizer_group_sizes": [
                        sum(parameter.numel() for parameter in group["params"])
                        for group in optimizer.param_groups
                    ],
                    "style_augmentation": style_record,
                }
            ),
            flush=True,
        )
    precision = cfg.train.get("precision", "fp16")
    scaler = make_grad_scaler(device, precision)
    if resume_state is not None:
        if resume_state.get("optimizer"):
            optimizer.load_state_dict(resume_state["optimizer"])
        if resume_state.get("scaler"):
            scaler.load_state_dict(resume_state["scaler"])
        if resume_state.get("rng") and not distributed:
            _restore_rng_state(resume_state["rng"])
    selection_metric, higher_is_better = checkpoint_selection(cfg.train)
    if resume_state is not None:
        fallback = -float("inf") if higher_is_better else float("inf")
        best = float(resume_state.get("best_selection_score", fallback))
        # Checkpoints written before configurable selection only recorded the
        # validation loss, so they remain resumable under the default policy.
        if "best_selection_score" not in resume_state and selection_metric == "val_loss":
            best = float(resume_state.get("best_val_loss", float("inf")))
    else:
        best = -float("inf") if higher_is_better else float("inf")
    if rank == 0 and counterfactual["enabled"]:
        print(json.dumps({"paired_counterfactual": counterfactual}), flush=True)
    if rank == 0 and scale_equivariance["enabled"]:
        print(json.dumps({"paired_scale_equivariance": scale_equivariance}), flush=True)
    if rank == 0 and anatomy_field["enabled"]:
        print(json.dumps({"scale_normalized_anatomy_field": anatomy_field}), flush=True)
    if rank == 0 and soft_topology["enabled"]:
        print(json.dumps({"soft_topology": soft_topology}), flush=True)
    accum = max(int(cfg.train.get("grad_accum_steps", 1)), 1)
    total_epochs = int(cfg.train.get("epochs", 2))
    for epoch in range(start_epoch, total_epochs):
        lr_scale = lr_multiplier(cfg.train, epoch, total_epochs)
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr * lr_scale
        learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
        scale_consistency_ramp = auxiliary_loss_warmup(
            epoch, scale_equivariance["warmup_epochs"]
        )
        anatomy_field_ramp = auxiliary_loss_warmup(
            epoch, anatomy_field["warmup_epochs"]
        )
        soft_topology_ramp = delayed_auxiliary_loss_ramp(
            epoch,
            start_epoch=soft_topology["start_epoch"],
            ramp_epochs=soft_topology["ramp_epochs"],
        )
        if distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        if trainable_scope == "global_classifier":
            # The residual classifier learns on deterministic frozen features;
            # keep decoder dropout disabled so its inherited source logit is a
            # stable baseline rather than augmentation-like noise.
            base.backbone.eval()
            base.decoder.eval()
            base.classifier.eval()
            assert base.global_classifier is not None
            base.global_classifier.train()
        elif cfg.model.get("freeze_backbone", True):
            base.backbone.eval()
        optimizer.zero_grad(set_to_none=True)
        running_sum = 0.0
        running_count = 0
        counterfactual_sums = {
            "clean_supervised": 0.0,
            "style_supervised": 0.0,
            "seg_consistency": 0.0,
            "cls_consistency": 0.0,
            "boundary_consistency": 0.0,
            "vcdr_consistency": 0.0,
            "rim_consistency": 0.0,
            "clinical_consistency": 0.0,
            "weighted_consistency": 0.0,
        }
        scale_equivariance_sums = {
            "clean_supervised": 0.0,
            "global_supervised": 0.0,
            "seg_equivariance": 0.0,
            "cls_invariance": 0.0,
            "vcdr_invariance": 0.0,
            "rim_invariance": 0.0,
            "clinical_invariance": 0.0,
            "weighted_consistency": 0.0,
            "sampled_scale": 0.0,
            "consistency_warmup": 0.0,
        }
        anatomy_field_sums = {
            "clean_field_supervised": 0.0,
            "global_field_supervised": 0.0,
            "field_equivariance": 0.0,
            "weighted_field": 0.0,
            "field_mask_consistency": 0.0,
            "weighted_field_mask_consistency": 0.0,
            "warmup": 0.0,
        }
        soft_topology_sums = {
            "topk_probability_excess": 0.0,
            "mean_violating_probability_excess": 0.0,
            "violating_pixel_fraction": 0.0,
            "violating_image_fraction": 0.0,
            "active_topk_fraction": 0.0,
            "weighted_topology": 0.0,
            "warmup": 0.0,
        }
        block_start = 0
        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            seg_target = (
                batch["seg_target"].to(device, non_blocking=True)
                if "seg_target" in batch
                else None
            )
            cls_target = (
                batch["cls_target"].to(device, non_blocking=True)
                if "cls_target" in batch
                else None
            )
            vcdr_target = (
                batch["vcdr_target"].to(device, non_blocking=True)
                if "vcdr_target" in batch
                else None
            )
            anatomy_field_target = (
                batch["anatomy_field_target"].to(device, non_blocking=True)
                if anatomy_field["enabled"] and "anatomy_field_target" in batch
                else None
            )
            vertical_geometry_target = (
                batch["vertical_geometry_target"].to(device, non_blocking=True)
                if vra_enabled and "vertical_geometry_target" in batch
                else None
            )
            style_images = (
                camera_style_counterfactual(images, counterfactual["style"])
                if counterfactual["enabled"]
                else None
            )
            geometry_batch = (
                scale_canvas_batch(images, seg_target, scale_equivariance)
                if scale_equivariance["enabled"]
                else None
            )
            with autocast_context(device, precision):
                output = model(images)
                clean_loss, _ = source_loss(
                    output.seg_logits,
                    output.cls_logits,
                    seg_target,
                    cls_target,
                    boundary_logits=output.boundary_logits,
                    soft_vcdr=output.soft_vcdr,
                    vcdr_target=vcdr_target,
                    **loss_kwargs,
                )
                raw_loss = clean_loss
                clean_vra_loss = None
                if vra_enabled:
                    if output.vertical_geometry_logits is None or vertical_geometry_target is None:
                        raise RuntimeError("VRA model or batch is missing vertical geometry targets")
                    clean_vra_loss = vertical_geometry_loss(
                        output.vertical_geometry_logits,
                        vertical_geometry_target,
                        log_margin_weight=vra_log_margin_weight,
                    )
                    raw_loss = raw_loss + vra_weight * clean_vra_loss
                hierarchy_outputs = [output]
                counterfactual_batch_stats: dict[str, float] | None = None
                scale_equivariance_batch_stats: dict[str, float] | None = None
                anatomy_field_batch_stats: dict[str, float] | None = None
                soft_topology_batch_stats: dict[str, float] | None = None
                if style_images is not None:
                    style_output = model(style_images)
                    hierarchy_outputs.append(style_output)
                    style_loss, _ = source_loss(
                        style_output.seg_logits,
                        style_output.cls_logits,
                        seg_target,
                        cls_target,
                        boundary_logits=style_output.boundary_logits,
                        soft_vcdr=style_output.soft_vcdr,
                        vcdr_target=vcdr_target,
                        **loss_kwargs,
                    )
                    consistency_loss, consistency_stats = paired_counterfactual_consistency_loss(
                        output,
                        style_output,
                        counterfactual["seg_consistency_weight"],
                        counterfactual["cls_consistency_weight"],
                        counterfactual["boundary_consistency_weight"],
                        counterfactual["clinical_consistency_weight"],
                    )
                    style_weight = counterfactual["style_supervision_weight"]
                    raw_loss = (
                        clean_loss + style_weight * style_loss
                    ) / (1.0 + style_weight) + consistency_loss
                    if vra_enabled:
                        if style_output.vertical_geometry_logits is None or clean_vra_loss is None:
                            raise RuntimeError("VRA style output is missing geometry logits")
                        style_vra_loss = vertical_geometry_loss(
                            style_output.vertical_geometry_logits,
                            vertical_geometry_target,
                            log_margin_weight=vra_log_margin_weight,
                        )
                        raw_loss = raw_loss + vra_weight * (
                            clean_vra_loss + style_weight * style_vra_loss
                        ) / (1.0 + style_weight)
                    counterfactual_batch_stats = {
                        "clean_supervised": float(clean_loss.detach()),
                        "style_supervised": float(style_loss.detach()),
                        **consistency_stats,
                    }
                elif geometry_batch is not None:
                    global_output = model(geometry_batch.images)
                    hierarchy_outputs.append(global_output)
                    global_loss, _ = source_loss(
                        global_output.seg_logits,
                        global_output.cls_logits,
                        geometry_batch.seg_target,
                        cls_target,
                        boundary_logits=global_output.boundary_logits,
                        soft_vcdr=global_output.soft_vcdr,
                        vcdr_target=None,
                        **loss_kwargs,
                    )
                    consistency_loss, consistency_stats = (
                        paired_scale_equivariance_consistency_loss(
                            output,
                            global_output,
                            geometry_batch.grid,
                            seg_weight=scale_equivariance[
                                "seg_consistency_weight"
                            ]
                            * scale_consistency_ramp,
                            cls_weight=scale_equivariance[
                                "cls_consistency_weight"
                            ]
                            * scale_consistency_ramp,
                            clinical_weight=scale_equivariance[
                                "clinical_consistency_weight"
                            ]
                            * scale_consistency_ramp,
                        )
                    )
                    global_weight = scale_equivariance[
                        "global_supervision_weight"
                    ]
                    raw_loss = (
                        clean_loss + global_weight * global_loss
                    ) / (1.0 + global_weight) + consistency_loss
                    if vra_enabled:
                        if global_output.vertical_geometry_logits is None or clean_vra_loss is None:
                            raise RuntimeError("VRA global output is missing geometry logits")
                        global_geometry_target = make_vertical_target(geometry_batch.seg_target)
                        global_vra_loss = vertical_geometry_loss(
                            global_output.vertical_geometry_logits,
                            global_geometry_target.to(device),
                            log_margin_weight=vra_log_margin_weight,
                        )
                        raw_loss = raw_loss + vra_weight * (
                            clean_vra_loss + global_weight * global_vra_loss
                        ) / (1.0 + global_weight)
                    if anatomy_field["enabled"]:
                        if anatomy_field_target is None:
                            raise RuntimeError("DAFL train batch is missing anatomy_field_target")
                        if (
                            output.anatomy_field_logits is None
                            or global_output.anatomy_field_logits is None
                        ):
                            raise RuntimeError("DAFL model is missing anatomy_field_logits")
                        field_loss, anatomy_field_batch_stats = (
                            paired_scale_anatomy_field_loss(
                                output.anatomy_field_logits,
                                global_output.anatomy_field_logits,
                                anatomy_field_target,
                                geometry_batch.grid,
                                supervision_weight=anatomy_field["supervision_weight"]
                                * anatomy_field_ramp,
                                equivariance_weight=anatomy_field[
                                    "equivariance_weight"
                                ]
                                * anatomy_field_ramp,
                                boundary_band=anatomy_field["boundary_band"],
                            )
                        )
                        mask_consistency = 0.5 * (
                            field_mask_consistency_loss(
                                output.seg_logits,
                                output.anatomy_field_logits,
                                temperature=anatomy_field["mask_consistency_temperature"],
                            )
                            + field_mask_consistency_loss(
                                global_output.seg_logits,
                                global_output.anatomy_field_logits,
                                temperature=anatomy_field["mask_consistency_temperature"],
                            )
                        )
                        weighted_mask_consistency = (
                            anatomy_field_ramp
                            * anatomy_field["mask_consistency_weight"]
                            * mask_consistency
                        )
                        field_loss = field_loss + weighted_mask_consistency
                        anatomy_field_batch_stats.update(
                            field_mask_consistency=float(mask_consistency.detach()),
                            weighted_field_mask_consistency=float(
                                weighted_mask_consistency.detach()
                            ),
                            warmup=float(anatomy_field_ramp),
                        )
                        raw_loss = raw_loss + field_loss
                    scale_equivariance_batch_stats = {
                        "clean_supervised": float(clean_loss.detach()),
                        "global_supervised": float(global_loss.detach()),
                        **consistency_stats,
                        "sampled_scale": float(
                            geometry_batch.scales.detach().mean()
                        ),
                        "consistency_warmup": float(scale_consistency_ramp),
                    }
                if soft_topology["enabled"]:
                    topology_terms = [
                        thresholded_topology_loss(
                            hierarchy_output.seg_logits,
                            threshold=soft_topology["threshold"],
                            topk_fraction=soft_topology["topk_fraction"],
                            gradient_target=soft_topology["gradient_target"],
                        )
                        for hierarchy_output in hierarchy_outputs
                    ]
                    topology_loss = torch.stack(
                        [term_loss for term_loss, _ in topology_terms]
                    ).mean()
                    weighted_topology = (
                        soft_topology_ramp
                        * soft_topology["weight"]
                        * topology_loss
                    )
                    raw_loss = raw_loss + weighted_topology
                    stat_names = topology_terms[0][1].keys()
                    soft_topology_batch_stats = {
                        name: sum(stats[name] for _, stats in topology_terms)
                        / len(topology_terms)
                        for name in stat_names
                    }
                    soft_topology_batch_stats.update(
                        weighted_topology=float(weighted_topology.detach()),
                        warmup=float(soft_topology_ramp),
                    )
                loss = raw_loss / accum
            scaler.scale(loss).backward()
            batch_size = int(images.shape[0])
            running_sum += float(raw_loss.detach()) * batch_size
            running_count += batch_size
            if counterfactual_batch_stats is not None:
                for name, value in counterfactual_batch_stats.items():
                    counterfactual_sums[name] += value * batch_size
            if scale_equivariance_batch_stats is not None:
                for name, value in scale_equivariance_batch_stats.items():
                    scale_equivariance_sums[name] += value * batch_size
            if anatomy_field_batch_stats is not None:
                for name, value in anatomy_field_batch_stats.items():
                    anatomy_field_sums[name] += value * batch_size
            if soft_topology_batch_stats is not None:
                for name, value in soft_topology_batch_stats.items():
                    soft_topology_sums[name] += value * batch_size
            is_boundary = ((step + 1) % accum == 0) or (step + 1 == len(train_loader))
            if is_boundary:
                actual_block = step - block_start + 1
                if actual_block != accum:
                    correction = float(accum) / float(actual_block)
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.get("grad_clip", 1.0))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                block_start = step + 1
        reduced_sum, reduced_count = _all_reduce_sum_count(running_sum, running_count, device, distributed)
        reduced_counterfactual: dict[str, float] | None = None
        if counterfactual["enabled"]:
            reduced_counterfactual = {}
            for name, total in counterfactual_sums.items():
                reduced_total, component_count = _all_reduce_sum_count(
                    total, running_count, device, distributed
                )
                reduced_counterfactual[name] = reduced_total / max(component_count, 1)
        reduced_scale_equivariance: dict[str, float] | None = None
        if scale_equivariance["enabled"]:
            reduced_scale_equivariance = {}
            for name, total in scale_equivariance_sums.items():
                reduced_total, component_count = _all_reduce_sum_count(
                    total, running_count, device, distributed
                )
                reduced_scale_equivariance[name] = reduced_total / max(
                    component_count, 1
                )
        reduced_anatomy_field: dict[str, float] | None = None
        if anatomy_field["enabled"]:
            reduced_anatomy_field = {}
            for name, total in anatomy_field_sums.items():
                reduced_total, component_count = _all_reduce_sum_count(
                    total, running_count, device, distributed
                )
                reduced_anatomy_field[name] = reduced_total / max(component_count, 1)
        reduced_soft_topology: dict[str, float] | None = None
        if soft_topology["enabled"]:
            reduced_soft_topology = {}
            for name, total in soft_topology_sums.items():
                reduced_total, component_count = _all_reduce_sum_count(
                    total, running_count, device, distributed
                )
                reduced_soft_topology[name] = reduced_total / max(component_count, 1)

        # Validation is intentionally performed by rank 0 on the complete
        # patient-disjoint validation manifest, then broadcast through the
        # barrier before the next epoch.
        val_loss = float("inf")
        val_seg_dice = float("nan")
        val_auroc = float("nan")
        val_auprc = float("nan")
        if rank == 0:
            val_sum = 0.0
            val_count = 0
            val_dice_sum = 0.0
            val_dice_count = 0
            val_cls_logits: list[torch.Tensor] = []
            val_cls_targets: list[torch.Tensor] = []
            with torch.no_grad():
                base.eval()
                for batch in val_loader:
                    images = batch["image"].to(device)
                    output = base(images)
                    loss, _ = source_loss(
                        output.seg_logits,
                        output.cls_logits,
                        batch.get("seg_target", None).to(device) if "seg_target" in batch else None,
                        batch.get("cls_target", None).to(device) if "cls_target" in batch else None,
                        boundary_logits=output.boundary_logits,
                        soft_vcdr=output.soft_vcdr,
                        vcdr_target=batch.get("vcdr_target", None).to(device) if "vcdr_target" in batch else None,
                        **loss_kwargs,
                    )
                    n = int(images.shape[0])
                    val_sum += float(loss) * n
                    val_count += n
                    if "seg_target" in batch:
                        target = batch["seg_target"].to(device)
                        probability = torch.sigmoid(output.seg_logits.float())
                        dims = (2, 3)
                        intersection = (probability * target).sum(dim=dims)
                        denominator = probability.sum(dim=dims) + target.sum(dim=dims)
                        dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
                        val_dice_sum += float(dice.sum())
                        val_dice_count += int(dice.numel())
                    if "cls_target" in batch:
                        val_cls_logits.append(output.cls_logits.detach().float().cpu())
                        val_cls_targets.append(batch["cls_target"].detach().float().cpu())
                val_loss = val_sum / max(val_count, 1)
                val_seg_dice = val_dice_sum / max(val_dice_count, 1)
                cls_metrics = validation_classification_metrics(
                    val_cls_logits, val_cls_targets
                )
                val_auroc = float(cls_metrics["auroc"])
                val_auprc = float(cls_metrics["auprc"])
            selection_scores = {
                "val_loss": val_loss,
                "val_seg_dice": val_seg_dice,
                "val_auroc": val_auroc,
                "val_auprc": val_auprc,
            }
            selection_score = selection_scores[selection_metric]
            if not math.isfinite(selection_score):
                raise RuntimeError(
                    f"Non-finite checkpoint selection metric {selection_metric}: {selection_score}"
                )
            payload = {
                "epoch": epoch,
                "train_loss": reduced_sum / max(reduced_count, 1),
                "val_loss": val_loss,
                "val_seg_dice": val_seg_dice,
                "val_auroc": val_auroc,
                "val_auprc": val_auprc,
                "selection_metric": selection_metric,
                "selection_score": selection_score,
                "world_size": world,
                "seed": int(base_seed),
                "lr_scale": lr_scale,
                "learning_rates": learning_rates,
            }
            if reduced_counterfactual is not None:
                payload["train_counterfactual"] = reduced_counterfactual
            if reduced_scale_equivariance is not None:
                payload["train_scale_equivariance"] = reduced_scale_equivariance
            if reduced_anatomy_field is not None:
                payload["train_scale_normalized_anatomy_field"] = reduced_anatomy_field
            if reduced_soft_topology is not None:
                payload["train_soft_topology"] = reduced_soft_topology
            print(json.dumps(payload), flush=True)
            checkpoint_payload = {
                "model": base.state_dict(),
                "model_architecture": model_architecture_for_config(cfg),
                "config": cfg.__dict__,
                "epoch": epoch,
                "best_val_loss": val_loss,
                "best_selection_score": best,
                "selection_metric": selection_metric,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": _rng_state(),
                "metrics": payload,
            }
            better = selection_score > best if higher_is_better else selection_score < best
            if better:
                best = selection_score
                checkpoint_payload["best_selection_score"] = best
                torch.save(checkpoint_payload, out_dir / "best.pt")
            checkpoint_payload["best_selection_score"] = best
            torch.save(checkpoint_payload, out_dir / "last.pt")
        if distributed:
            torch.distributed.barrier()
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
