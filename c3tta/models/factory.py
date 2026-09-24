"""Model construction shared by training, evaluation, and TTA entry points."""

from __future__ import annotations

from c3tta.models.backbones import build_backbone
from c3tta.models.multitask import (
    CONDITIONAL_CUP_HIERARCHY,
    CUP_PRESERVING_HIERARCHY,
    INDEPENDENT_SEGMENTATION,
    C3MultiTask,
    resolve_segmentation_hierarchy_mode,
)
from c3tta.models.style_augmentation import build_style_augmentation, style_method_from_config


# Keep the public constant as the legacy architecture marker: existing tools
# and source checkpoints use it.  New checkpoints derive their marker from
# their model configuration below.
MODEL_ARCHITECTURE = "c3tta-adapter-fused-classifier-v2"
PYRAMID_FPN_MODEL_ARCHITECTURE = "c3tta-pyramid-fpn-multitask-v1"
PYRAMID_FPN_LORA_MODEL_ARCHITECTURE = "c3tta-pyramid-fpn-lora-multitask-v1"
PYRAMID_FPN_LORA_GLOBAL_CLASSIFIER_MODEL_ARCHITECTURE = (
    "c3tta-pyramid-fpn-lora-global-classifier-v1"
)
PYRAMID_FPN_LORA_CROSS_TASK_TTA_MODEL_ARCHITECTURE = (
    "c3tta-pyramid-fpn-lora-cross-task-tta-v1"
)
RETINAL_FOUNDATION_BOUNDARY_MODEL_ARCHITECTURE = (
    "c3tta-retinal-foundation-boundary-cross-task-v1"
)
RETINAL_FOUNDATION_BOUNDARY_DETACH_MODEL_ARCHITECTURE = (
    "c3tta-retinal-foundation-boundary-cross-task-detach-v1"
)
PYRAMID_FPN_LORA_STYLE_DG_MODEL_ARCHITECTURE = (
    "c3tta-pyramid-fpn-lora-style-dg-v1"
)
PYRAMID_FPN_LORA_DAFL_MODEL_ARCHITECTURE = (
    "c3tta-pyramid-fpn-lora-dafl-v1"
)
PYRAMID_FPN_LORA_GEOMETRY_COUPLED_DAFL_MODEL_ARCHITECTURE = (
    "c3tta-pyramid-fpn-lora-geometry-coupled-dafl-v1"
)
PYRAMID_FPN_LORA_CUP_PRESERVING_DAFL_MODEL_ARCHITECTURE = (
    "c3tta-pyramid-fpn-lora-cup-preserving-dafl-v2"
)
PYRAMID_FPN_LORA_CUP_PRESERVING_NO_DETACH_DAFL_MODEL_ARCHITECTURE = (
    "c3tta-pyramid-fpn-lora-cup-preserving-no-detach-dafl-v2"
)
PYRAMID_FPN_LORA_CUP_PRESERVING_VRA_MODEL_ARCHITECTURE = (
    "c3tta-pyramid-fpn-lora-cup-preserving-vra-v1"
)


def decoder_type_for_config(cfg) -> str:
    raw = str(cfg.model.get("decoder_type", "adapter_fusion")).lower()
    aliases = {
        "adapter_fusion": "adapter_fusion",
        "feature_fusion": "adapter_fusion",
        "mean_fusion": "adapter_fusion",
        "pyramid_fpn": "pyramid_fpn",
        "upernet": "pyramid_fpn",
    }
    try:
        return aliases[raw]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported decoder_type {raw!r}; choose adapter_fusion or pyramid_fpn"
        ) from exc


def segmentation_hierarchy_for_config(cfg) -> str:
    return resolve_segmentation_hierarchy_mode(
        cfg.model.get("segmentation_hierarchy"),
        hierarchical_segmentation=cfg.model.get("hierarchical_segmentation"),
    )


def model_architecture_for_config(cfg) -> str:
    decoder_type = decoder_type_for_config(cfg)
    style_method = style_method_from_config(cfg.train.get("style_augmentation"))
    segmentation_hierarchy = segmentation_hierarchy_for_config(cfg)
    vertical_geometry_head = bool(cfg.model.get("vertical_geometry_head", False))
    if vertical_geometry_head:
        if decoder_type != "pyramid_fpn" or not bool(cfg.model.get("anatomy_field_head", False)):
            raise ValueError(
                "vertical_geometry_head requires the pyramid-FPN anatomy-field profile"
            )
        if segmentation_hierarchy == CUP_PRESERVING_HIERARCHY:
            return PYRAMID_FPN_LORA_CUP_PRESERVING_VRA_MODEL_ARCHITECTURE
        raise ValueError(
            "vertical_geometry_head requires cup_preserving hierarchy"
        )
    detach_cup_from_od = cfg.model.get("detach_cup_from_od", True)
    if not isinstance(detach_cup_from_od, bool):
        raise ValueError("model.detach_cup_from_od must be a boolean")
    if (
        not detach_cup_from_od
        and segmentation_hierarchy != CUP_PRESERVING_HIERARCHY
    ):
        raise ValueError(
            "model.detach_cup_from_od=false requires "
            "model.segmentation_hierarchy=cup_preserving"
        )
    if decoder_type == "adapter_fusion":
        if style_method != "none":
            raise ValueError("Style augmentation requires decoder_type=pyramid_fpn")
        if segmentation_hierarchy != INDEPENDENT_SEGMENTATION:
            raise ValueError(
                "hierarchical_segmentation requires decoder_type=pyramid_fpn"
            )
        return MODEL_ARCHITECTURE
    if decoder_type == "pyramid_fpn":
        anatomy_field_head = bool(cfg.model.get("anatomy_field_head", False))
        if segmentation_hierarchy != INDEPENDENT_SEGMENTATION and not anatomy_field_head:
            raise ValueError(
                "hierarchical_segmentation requires the anatomy-field DAFL model profile"
            )
        if anatomy_field_head:
            if style_method != "none":
                raise ValueError("DAFL feasibility profile cannot combine style augmentation")
            if (
                int(cfg.model.get("lora_rank", 0)) <= 0
                and str(cfg.model.get("backbone", "tiny")).lower() != "tiny"
            ):
                raise ValueError("DAFL feasibility profile requires a LoRA-enabled pyramid FPN")
            if segmentation_hierarchy == CONDITIONAL_CUP_HIERARCHY:
                return PYRAMID_FPN_LORA_GEOMETRY_COUPLED_DAFL_MODEL_ARCHITECTURE
            if segmentation_hierarchy == CUP_PRESERVING_HIERARCHY:
                if not detach_cup_from_od:
                    return (
                        PYRAMID_FPN_LORA_CUP_PRESERVING_NO_DETACH_DAFL_MODEL_ARCHITECTURE
                    )
                return PYRAMID_FPN_LORA_CUP_PRESERVING_DAFL_MODEL_ARCHITECTURE
            return PYRAMID_FPN_LORA_DAFL_MODEL_ARCHITECTURE
        if style_method != "none":
            incompatible = (
                bool(cfg.model.get("boundary_aware", False))
                or bool(cfg.model.get("region_conditioned_classifier", False))
                or bool(cfg.model.get("global_residual_classifier", False))
                or int(cfg.model.get("tta_adapter_rank", 0)) > 0
            )
            if incompatible:
                raise ValueError(
                    "The frozen Style-DG baselines use the Plain/ERM task profile; "
                    "do not combine them with boundary, cross-task, global-classifier, "
                    "or online-TTA modules."
                )
            if (
                int(cfg.model.get("lora_rank", 0)) <= 0
                and str(cfg.model.get("backbone", "tiny")).lower() != "tiny"
            ):
                raise ValueError("Style-DG baseline requires a LoRA-enabled Pyramid-FPN")
            return PYRAMID_FPN_LORA_STYLE_DG_MODEL_ARCHITECTURE
        boundary_aware = bool(cfg.model.get("boundary_aware", False))
        region_conditioned_classifier = bool(
            cfg.model.get("region_conditioned_classifier", False)
        )
        detach_region_masks = bool(cfg.model.get("detach_region_masks", False))
        if detach_region_masks and not region_conditioned_classifier:
            raise ValueError(
                "detach_region_masks requires region_conditioned_classifier=true"
            )
        if boundary_aware or region_conditioned_classifier:
            # The production retinal profile requires a LoRA-enabled
            # foundation backbone. The built-in tiny backbone is the sole
            # exception so CPU/DDP smoke tests can exercise the same heads and
            # losses without downloading or allocating a ViT-L checkpoint.
            if (
                int(cfg.model.get("lora_rank", 0)) <= 0
                and str(cfg.model.get("backbone", "tiny")).lower() != "tiny"
            ):
                raise ValueError(
                    "retinal foundation boundary profile requires LoRA-enabled backbone"
                )
            if int(cfg.model.get("tta_adapter_rank", 0)) > 0:
                raise ValueError(
                    "boundary cross-task source profile and online TTA profile are separate"
                )
            if detach_region_masks:
                return RETINAL_FOUNDATION_BOUNDARY_DETACH_MODEL_ARCHITECTURE
            return RETINAL_FOUNDATION_BOUNDARY_MODEL_ARCHITECTURE
        if int(cfg.model.get("tta_adapter_rank", 0)) > 0:
            if bool(cfg.model.get("global_residual_classifier", False)):
                raise ValueError(
                    "Pyramid-FPN cross-task TTA and global_residual_classifier are separate "
                    "profiles; disable the global residual classifier for the unified model"
                )
            if int(cfg.model.get("lora_rank", 0)) <= 0:
                raise ValueError("Pyramid-FPN cross-task TTA profile requires LoRA")
            return PYRAMID_FPN_LORA_CROSS_TASK_TTA_MODEL_ARCHITECTURE
        if bool(cfg.model.get("global_residual_classifier", False)):
            if int(cfg.model.get("lora_rank", 0)) <= 0:
                raise ValueError("global residual classifier profile requires LoRA-enabled pyramid FPN")
            return PYRAMID_FPN_LORA_GLOBAL_CLASSIFIER_MODEL_ARCHITECTURE
        if int(cfg.model.get("lora_rank", 0)) > 0:
            return PYRAMID_FPN_LORA_MODEL_ARCHITECTURE
        return PYRAMID_FPN_MODEL_ARCHITECTURE
    raise AssertionError(f"missing architecture marker for {decoder_type}")


def validate_checkpoint_architecture(checkpoint: dict, cfg=None) -> None:
    found = checkpoint.get("model_architecture")
    supported = {
        MODEL_ARCHITECTURE,
        PYRAMID_FPN_MODEL_ARCHITECTURE,
        PYRAMID_FPN_LORA_MODEL_ARCHITECTURE,
        PYRAMID_FPN_LORA_GLOBAL_CLASSIFIER_MODEL_ARCHITECTURE,
        PYRAMID_FPN_LORA_CROSS_TASK_TTA_MODEL_ARCHITECTURE,
        RETINAL_FOUNDATION_BOUNDARY_MODEL_ARCHITECTURE,
        RETINAL_FOUNDATION_BOUNDARY_DETACH_MODEL_ARCHITECTURE,
        PYRAMID_FPN_LORA_STYLE_DG_MODEL_ARCHITECTURE,
        PYRAMID_FPN_LORA_DAFL_MODEL_ARCHITECTURE,
        PYRAMID_FPN_LORA_GEOMETRY_COUPLED_DAFL_MODEL_ARCHITECTURE,
        PYRAMID_FPN_LORA_CUP_PRESERVING_DAFL_MODEL_ARCHITECTURE,
        PYRAMID_FPN_LORA_CUP_PRESERVING_NO_DETACH_DAFL_MODEL_ARCHITECTURE,
        PYRAMID_FPN_LORA_CUP_PRESERVING_VRA_MODEL_ARCHITECTURE,
    }
    if found not in supported:
        found_description = found or "legacy checkpoint without architecture metadata"
        raise RuntimeError(
            f"Incompatible model architecture: found {found_description!r}. Supported "
            f"markers are {sorted(supported)!r}; legacy source checkpoints cannot be "
            "safely reused; retrain the source model with the current code."
        )
    if cfg is not None:
        expected = model_architecture_for_config(cfg)
        if found != expected:
            raise RuntimeError(
                f"Checkpoint architecture {found!r} does not match the requested "
                f"configuration {expected!r}. Use the checkpoint's saved model "
                "configuration or select a matching checkpoint."
            )


def validate_init_checkpoint_architecture(checkpoint: dict, cfg) -> tuple[str, ...]:
    """Allow only identity-preserving source-model expansions.

    Ordinary resume/evaluation requires an exact architecture marker.  The
    training-only ``--init-checkpoint`` path additionally supports the two
    historical zero-initialized expansions that add either a global
    classifier or a shared TTA adapter.  Hierarchy modes are intentionally not
    transferable: their state tensors have matching shapes but different
    channel semantics.
    """

    validate_checkpoint_architecture(checkpoint)
    found = checkpoint.get("model_architecture")
    expected = model_architecture_for_config(cfg)
    allowed_expansions = {
        (
            PYRAMID_FPN_LORA_MODEL_ARCHITECTURE,
            PYRAMID_FPN_LORA_GLOBAL_CLASSIFIER_MODEL_ARCHITECTURE,
        ): ("global_classifier.",),
        (
            PYRAMID_FPN_LORA_MODEL_ARCHITECTURE,
            PYRAMID_FPN_LORA_CROSS_TASK_TTA_MODEL_ARCHITECTURE,
        ): ("decoder.tta_adapter.",),
        (
            PYRAMID_FPN_LORA_CUP_PRESERVING_DAFL_MODEL_ARCHITECTURE,
            PYRAMID_FPN_LORA_CUP_PRESERVING_VRA_MODEL_ARCHITECTURE,
        ): ("decoder.vertical_geometry_head.", "vertical_geometry_alpha"),
    }
    if found == expected:
        return ()
    try:
        return allowed_expansions[(found, expected)]
    except KeyError as exc:
        raise RuntimeError(
            f"Init checkpoint architecture {found!r} cannot initialize requested "
            f"architecture {expected!r}. Only an exact match or an audited "
            "zero-initialized global-classifier/TTA-adapter expansion is allowed; "
            "cross-hierarchy initialization is unsafe."
        ) from exc


def build_model(cfg, initialize_backbone: bool = True) -> C3MultiTask:
    backbone = build_backbone(
        cfg.model.get("backbone", "tiny"),
        cfg.model.get("checkpoint") if initialize_backbone else None,
        cfg.model.get("feature_indices", (2, 5, 8, 11)),
        cfg.model.get("image_size", 448),
        pretrained=initialize_backbone,
    )
    lora_rank = int(cfg.model.get("lora_rank", 0))
    if lora_rank < 0:
        raise ValueError("lora_rank must be non-negative")
    if lora_rank:
        lora_setter = getattr(backbone, "enable_lora", None)
        if lora_setter is None:
            raise ValueError("LoRA requires a timm Transformer backbone, not this backbone")
        lora_setter(
            lora_rank,
            float(cfg.model.get("lora_alpha", lora_rank)),
            int(cfg.model.get("lora_last_blocks", 4)),
            tuple(cfg.model.get("lora_targets", ("attn.qkv",))),
            float(cfg.model.get("lora_dropout", 0.0)),
        )
    dim = getattr(backbone, "embed_dim", getattr(backbone, "num_features", 64))
    decoder_type = decoder_type_for_config(cfg)
    style_augmentation = build_style_augmentation(cfg.train.get("style_augmentation"))
    return C3MultiTask(
        backbone=backbone,
        backbone_dim=dim,
        levels=len(cfg.model.get("feature_indices", (2, 5, 8, 11))),
        decoder_channels=cfg.model.get("decoder_channels", 128),
        adapter_rank=cfg.model.get("adapter_rank", 32),
        classifier_hidden_dim=cfg.model.get("classifier_hidden_dim"),
        adapter_scale_init=cfg.model.get("adapter_scale_init", 1e-2),
        decoder_type=decoder_type,
        decoder_dropout=cfg.model.get("decoder_dropout", 0.1),
        ppm_bins=tuple(cfg.model.get("ppm_bins", (1, 2, 3, 6))),
        global_residual_classifier=bool(
            cfg.model.get("global_residual_classifier", False)
        ),
        tta_adapter_rank=int(cfg.model.get("tta_adapter_rank", 0)),
        tta_adapter_scale_init=float(
            cfg.model.get("tta_adapter_scale_init", 1e-2)
        ),
        boundary_aware=bool(cfg.model.get("boundary_aware", False)),
        anatomy_field_head=bool(cfg.model.get("anatomy_field_head", False)),
        region_conditioned_classifier=bool(
            cfg.model.get("region_conditioned_classifier", False)
        ),
        style_augmentation=style_augmentation,
        detach_region_masks=bool(cfg.model.get("detach_region_masks", False)),
        hierarchical_segmentation=cfg.model.get("hierarchical_segmentation"),
        segmentation_hierarchy=cfg.model.get("segmentation_hierarchy"),
        detach_cup_from_od=cfg.model.get("detach_cup_from_od", True),
        vertical_geometry_head=bool(cfg.model.get("vertical_geometry_head", False)),
        vertical_geometry_calibration_enabled=bool(
            cfg.model.get("vertical_geometry_calibration", False)
        ),
        vertical_geometry_alpha_init=float(
            cfg.model.get("vertical_geometry_alpha_init", 0.0)
        ),
        vertical_geometry_temperature=float(
            cfg.model.get("vertical_geometry_temperature", 0.02)
        ),
        vertical_geometry_kappa=float(cfg.model.get("vertical_geometry_kappa", 2.0)),
    )
