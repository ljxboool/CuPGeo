from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .backbones import BackboneOutput
from .decoder import FeatureFusionDecoder, PyramidFPNDecoder


INDEPENDENT_SEGMENTATION = "independent"
CONDITIONAL_CUP_HIERARCHY = "conditional_cup"
CUP_PRESERVING_HIERARCHY = "cup_preserving"
SEGMENTATION_HIERARCHY_MODES = frozenset(
    {
        INDEPENDENT_SEGMENTATION,
        CONDITIONAL_CUP_HIERARCHY,
        CUP_PRESERVING_HIERARCHY,
    }
)


@dataclass
class ModelOutput:
    seg_logits: torch.Tensor
    cls_logits: torch.Tensor
    soft_vcdr: torch.Tensor
    rim_proxy: torch.Tensor
    features: list[torch.Tensor] | None = None
    boundary_logits: torch.Tensor | None = None
    anatomy_field_logits: torch.Tensor | None = None
    vertical_geometry_logits: torch.Tensor | None = None


def _validate_od_oc_logits(raw_logits: torch.Tensor) -> torch.Tensor:
    if raw_logits.ndim != 4 or raw_logits.shape[1] != 2:
        raise ValueError("hierarchical OD/OC logits require shape [B, 2, H, W]")
    if not raw_logits.is_floating_point():
        raise ValueError("hierarchical OD/OC logits must be floating point")
    return raw_logits.float()


def _probabilities_to_finite_logits(probabilities: torch.Tensor) -> torch.Tensor:
    # Keep the logit representation finite for FP16 training and downstream
    # prediction serialization while preserving the probability order.
    return torch.logit(probabilities.clamp(min=1e-5, max=1.0 - 1e-5))


def resolve_segmentation_hierarchy_mode(
    hierarchy_mode: str | None,
    *,
    hierarchical_segmentation: bool | None = None,
) -> str:
    """Resolve an explicit hierarchy mode while preserving legacy configs.

    Historical Geometry-coupled checkpoints used only the boolean
    ``hierarchical_segmentation`` flag.  It maps to ``conditional_cup`` so the
    old model remains exactly reproducible.  New experiments should set the
    explicit mode and keep the boolean consistent for audit readability.
    """

    if hierarchical_segmentation is not None and not isinstance(
        hierarchical_segmentation, bool
    ):
        raise ValueError("model.hierarchical_segmentation must be a boolean")
    legacy_mode = (
        CONDITIONAL_CUP_HIERARCHY
        if hierarchical_segmentation is True
        else INDEPENDENT_SEGMENTATION
    )
    if hierarchy_mode is None:
        return legacy_mode
    normalized = str(hierarchy_mode).strip().lower().replace("-", "_")
    aliases = {
        "none": INDEPENDENT_SEGMENTATION,
        "independent": INDEPENDENT_SEGMENTATION,
        "legacy": CONDITIONAL_CUP_HIERARCHY,
        "hard": CONDITIONAL_CUP_HIERARCHY,
        "conditional_cup": CONDITIONAL_CUP_HIERARCHY,
        "cup_preserving": CUP_PRESERVING_HIERARCHY,
        "cup_preserving_disc": CUP_PRESERVING_HIERARCHY,
    }
    try:
        resolved = aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported segmentation_hierarchy {hierarchy_mode!r}; choose one of "
            f"{sorted(SEGMENTATION_HIERARCHY_MODES)!r}"
        ) from exc
    if (
        hierarchical_segmentation is not None
        and hierarchical_segmentation != (resolved != INDEPENDENT_SEGMENTATION)
    ):
        raise ValueError(
            "model.hierarchical_segmentation conflicts with "
            f"model.segmentation_hierarchy={resolved!r}"
        )
    return resolved


def hierarchy_constrained_od_oc_logits(raw_logits: torch.Tensor) -> torch.Tensor:
    """Legacy disc-gated cup logits satisfying ``p_OC <= p_OD``.

    The segmentation head still predicts two unrestricted channels.  The
    second channel is interpreted as the cup probability *conditional* on the
    optic-disc probability.  This function is intentionally unchanged for
    exact reproduction of Geometry-coupled DAFL v1.
    """

    raw = _validate_od_oc_logits(raw_logits)
    od_probability = torch.sigmoid(raw[:, 0:1])
    oc_probability = od_probability * torch.sigmoid(raw[:, 1:2])
    probabilities = torch.cat((od_probability, oc_probability), dim=1)
    return _probabilities_to_finite_logits(probabilities)


def cup_preserving_od_oc_logits(
    raw_logits: torch.Tensor,
    *,
    detach_cup_from_od: bool = True,
) -> torch.Tensor:
    """Strict hierarchy that isolates the cup logit from direct OD gradients.

    The second raw channel remains the marginal cup probability.  The first
    channel predicts residual rim occupancy conditioned on a pixel not already
    belonging to the cup::

        p_OC = sigmoid(z_OC)
        p_OD = p_OC + (1 - p_OC) * sigmoid(z_rim)

    This stick-breaking form represents background, rim, and cup with two
    logits and guarantees ``p_OC <= p_OD`` at every threshold.  By default the
    cup logit is detached only in the OD branch: the forward probabilities are
    unchanged, but OD supervision cannot move the cup boundary.  OC
    supervision and shared-encoder gradients remain fully trainable.

    The equivalent OD logit is evaluated directly as
    ``log(exp(z_OC) + exp(z_rim) + exp(z_OC + z_rim))``.  This avoids the
    unstable ``sigmoid -> clamp -> logit`` round trip under mixed precision.
    """

    raw = _validate_od_oc_logits(raw_logits)
    rim_logit = raw[:, 0:1]
    oc_logit = raw[:, 1:2]
    oc_logit_for_od = oc_logit.detach() if detach_cup_from_od else oc_logit
    od_logit = torch.logsumexp(
        torch.stack(
            (
                oc_logit_for_od,
                rim_logit,
                oc_logit_for_od + rim_logit,
            ),
            dim=0,
        ),
        dim=0,
    )
    return torch.cat((od_logit, oc_logit), dim=1)


def vertical_geometry_calibration(
    raw_logits: torch.Tensor,
    geometry_logits: torch.Tensor,
    hierarchy: str,
    *,
    alpha: torch.Tensor | float,
    temperature: float = 0.02,
    kappa: float = 2.0,
) -> torch.Tensor:
    """Apply a bounded vertical cup prior before hierarchical composition."""

    if raw_logits.ndim != 4 or geometry_logits.ndim != 2 or geometry_logits.shape[1] != 5:
        raise ValueError("invalid vertical geometry calibration shapes")
    if temperature <= 0.0 or kappa <= 0.0:
        raise ValueError("calibration temperature and kappa must be positive")
    if hierarchy == CUP_PRESERVING_HIERARCHY:
        cup_index = 1
        cup_probability = torch.sigmoid(raw_logits[:, cup_index : cup_index + 1].float())
    else:
        raise ValueError("vertical rim allocation requires cup-preserving hierarchy")
    lengths = torch.softmax(geometry_logits.float(), dim=1)
    top_cup = lengths[:, 0] + lengths[:, 1]
    bottom_cup = top_cup + lengths[:, 2]
    height = raw_logits.shape[-2]
    y = torch.linspace(0.0, 1.0, height, device=raw_logits.device, dtype=torch.float32)
    y = y.view(1, 1, height, 1)
    prior_probability = torch.sigmoid((y - top_cup[:, None, None, None]) / temperature)
    prior_probability = prior_probability * torch.sigmoid(
        (bottom_cup[:, None, None, None] - y) / temperature
    )
    prior = torch.logit(prior_probability.clamp(1e-3, 1.0 - 1e-3)).clamp(-kappa, kappa)
    gate = 4.0 * cup_probability * (1.0 - cup_probability)
    calibrated = raw_logits.clone()
    calibrated[:, cup_index : cup_index + 1] = (
        raw_logits[:, cup_index : cup_index + 1].float()
        + alpha * gate * prior
    ).to(dtype=raw_logits.dtype)
    return calibrated


def _soft_vertical_diameter(mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # mask: [B,1,H,W]; differentiable weighted second moment along y.
    b, _, h, _ = mask.shape
    y = torch.linspace(-1.0, 1.0, h, device=mask.device, dtype=mask.dtype).view(1, 1, h, 1)
    mass_y = mask.sum(dim=3)
    mass = mass_y.sum(dim=2).clamp_min(eps)
    mean = (mass_y * y.squeeze(-1)).sum(dim=2) / mass
    var = (mass_y * (y.squeeze(-1) - mean.unsqueeze(-1)) ** 2).sum(dim=2) / mass
    return 4.0 * torch.sqrt(var.clamp_min(eps))


def model_output_from_logits(
    seg_logits: torch.Tensor,
    cls_logits: torch.Tensor,
    features: list[torch.Tensor] | None = None,
    boundary_logits: torch.Tensor | None = None,
    anatomy_field_logits: torch.Tensor | None = None,
    vertical_geometry_logits: torch.Tensor | None = None,
) -> ModelOutput:
    """Construct a complete prediction after label-free logit ensembling.

    Segmentation inference may average predictions across geometric views or
    compatible source models.  The clinically coupled quantities must then be
    recomputed from the fused logits instead of copied from an individual
    view, otherwise the artifact would contain internally inconsistent tasks.
    """
    probs = torch.sigmoid(seg_logits)
    od, oc = probs[:, 0:1], probs[:, 1:2]
    soft_vcdr = _soft_vertical_diameter(oc) / (_soft_vertical_diameter(od) + 1e-6)
    rim_proxy = (od.sum(dim=(-2, -1)) - oc.sum(dim=(-2, -1))) / (
        od.sum(dim=(-2, -1)) + 1e-6
    )
    return ModelOutput(
        seg_logits=seg_logits,
        cls_logits=cls_logits,
        soft_vcdr=soft_vcdr,
        rim_proxy=rim_proxy,
        features=features,
        boundary_logits=boundary_logits,
        anatomy_field_logits=anatomy_field_logits,
        vertical_geometry_logits=vertical_geometry_logits,
    )


class C3MultiTask(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        backbone_dim: int,
        levels: int = 4,
        decoder_channels: int = 128,
        adapter_rank: int = 32,
        classifier_hidden_dim: int | None = None,
        adapter_scale_init: float = 1e-2,
        decoder_type: str = "adapter_fusion",
        decoder_dropout: float = 0.1,
        ppm_bins: tuple[int, ...] = (1, 2, 3, 6),
        global_residual_classifier: bool = False,
        tta_adapter_rank: int = 0,
        tta_adapter_scale_init: float = 1e-2,
        boundary_aware: bool = False,
        anatomy_field_head: bool = False,
        region_conditioned_classifier: bool = False,
        style_augmentation: nn.Module | None = None,
        detach_region_masks: bool = False,
        hierarchical_segmentation: bool | None = None,
        segmentation_hierarchy: str | None = None,
        detach_cup_from_od: bool = True,
        vertical_geometry_head: bool = False,
        vertical_geometry_calibration_enabled: bool = False,
        vertical_geometry_alpha_init: float = 0.0,
        vertical_geometry_temperature: float = 0.02,
        vertical_geometry_kappa: float = 2.0,
    ) -> None:
        super().__init__()
        if classifier_hidden_dim is None:
            classifier_hidden_dim = decoder_channels
        if decoder_channels <= 0 or classifier_hidden_dim <= 0:
            raise ValueError("decoder and classifier channel dimensions must be positive")
        self.backbone = backbone
        self.decoder_type = decoder_type
        self.region_conditioned_classifier = bool(region_conditioned_classifier)
        self.detach_region_masks = bool(detach_region_masks)
        self.segmentation_hierarchy = resolve_segmentation_hierarchy_mode(
            segmentation_hierarchy,
            hierarchical_segmentation=hierarchical_segmentation,
        )
        if not isinstance(detach_cup_from_od, bool):
            raise ValueError("model.detach_cup_from_od must be a boolean")
        self.detach_cup_from_od = detach_cup_from_od
        self.vertical_geometry_calibration_enabled = bool(
            vertical_geometry_calibration_enabled
        )
        self.vertical_geometry_temperature = float(vertical_geometry_temperature)
        self.vertical_geometry_kappa = float(vertical_geometry_kappa)
        if self.vertical_geometry_calibration_enabled and not vertical_geometry_head:
            raise ValueError("vertical geometry calibration requires its prediction head")
        self.hierarchical_segmentation = (
            self.segmentation_hierarchy != INDEPENDENT_SEGMENTATION
        )
        if self.detach_region_masks and not self.region_conditioned_classifier:
            raise ValueError(
                "detach_region_masks requires region_conditioned_classifier=true"
            )
        if self.hierarchical_segmentation and not anatomy_field_head:
            raise ValueError(
                "hierarchical_segmentation is reserved for the anatomy-field DAFL profile"
            )
        if self.hierarchical_segmentation and decoder_type != "pyramid_fpn":
            raise ValueError(
                "hierarchical_segmentation requires decoder_type=pyramid_fpn"
            )
        if (
            not self.detach_cup_from_od
            and self.segmentation_hierarchy != CUP_PRESERVING_HIERARCHY
        ):
            raise ValueError(
                "model.detach_cup_from_od=false requires "
                "model.segmentation_hierarchy=cup_preserving"
            )
        segmentation_out_channels = 2
        if decoder_type == "adapter_fusion":
            if style_augmentation is not None:
                raise ValueError(
                    "Style augmentation is only defined for the shared Pyramid-FPN feature"
                )
            self.decoder = FeatureFusionDecoder(
                backbone_dim,
                levels,
                decoder_channels,
                adapter_rank,
                segmentation_out_channels,
                adapter_scale_init=adapter_scale_init,
            )
        elif decoder_type == "pyramid_fpn":
            self.decoder = PyramidFPNDecoder(
                backbone_dim,
                levels,
                decoder_channels,
                segmentation_out_channels,
                dropout=decoder_dropout,
                ppm_bins=ppm_bins,
                tta_adapter_rank=tta_adapter_rank,
                tta_adapter_scale_init=tta_adapter_scale_init,
                boundary_aware=boundary_aware,
                anatomy_field_head=anatomy_field_head,
                anatomy_field_out_channels=2,
                vertical_geometry_head=vertical_geometry_head,
                style_augmentation=style_augmentation,
            )
        else:
            raise ValueError(f"Unsupported decoder_type: {decoder_type!r}")
        if self.segmentation_hierarchy == CUP_PRESERVING_HIERARCHY:
            segmentation_head = getattr(self.decoder, "segmentation_head", None)
            if (
                not isinstance(segmentation_head, nn.Conv2d)
                or segmentation_head.bias is None
                or segmentation_head.bias.numel() != 2
            ):
                raise RuntimeError(
                    "cup-preserving hierarchy requires a two-channel Conv2d "
                    "segmentation head with bias"
                )
            # Match the legacy hard hierarchy's neutral initial marginals at
            # zero feature contribution: p_OC=0.25 and p_OD=0.50.  Without
            # this calibration, zero logits would start v2 at p_OD=0.75.
            with torch.no_grad():
                segmentation_head.bias.copy_(
                    segmentation_head.bias.new_tensor(
                        (math.log(0.5), math.log(1.0 / 3.0))
                    )
                )
        classifier_input_dim = decoder_channels * (
            3 if self.region_conditioned_classifier else 1
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(classifier_hidden_dim, 1),
        )
        # The dense FPN feature is excellent for OD/OC masks but can discard
        # global retinal context useful for glaucoma classification.  This
        # optional branch adds a frozen encoder-level descriptor as a residual
        # logit.  Its final projection starts at exactly zero, so transferring
        # an existing segmentation checkpoint preserves every prediction at
        # step zero and lets us optimize classification without perturbing
        # segmentation.
        self.global_classifier: nn.Module | None
        if global_residual_classifier:
            self.global_classifier = nn.Sequential(
                nn.LayerNorm(backbone_dim),
                nn.Linear(backbone_dim, 1),
            )
            final = self.global_classifier[-1]
            assert isinstance(final, nn.Linear)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        else:
            self.global_classifier = None
        self.vertical_geometry_head_enabled = bool(vertical_geometry_head)
        self.vertical_geometry_alpha: nn.Parameter | None
        if vertical_geometry_head:
            self.vertical_geometry_alpha = nn.Parameter(
                torch.tensor(float(vertical_geometry_alpha_init))
            )
        else:
            self.vertical_geometry_alpha = None

    def forward(self, images: torch.Tensor) -> ModelOutput:
        backbone_out: BackboneOutput = self.backbone(images)
        decoder_output = self.decoder(backbone_out.features, images.shape[-2:])
        seg_logits, adapted, fused = decoder_output[:3]
        boundary_logits = decoder_output[3] if len(decoder_output) >= 4 else None
        anatomy_field_logits = decoder_output[4] if len(decoder_output) >= 5 else None
        vertical_geometry_logits = decoder_output[5] if len(decoder_output) >= 6 else None
        if self.vertical_geometry_calibration_enabled:
            if vertical_geometry_logits is None or self.vertical_geometry_alpha is None:
                raise RuntimeError("vertical geometry calibration head is missing")
            raw_seg_logits = vertical_geometry_calibration(
                seg_logits,
                vertical_geometry_logits,
                self.segmentation_hierarchy,
                alpha=torch.tanh(self.vertical_geometry_alpha),
                temperature=self.vertical_geometry_temperature,
                kappa=self.vertical_geometry_kappa,
            )
            if self.segmentation_hierarchy == CUP_PRESERVING_HIERARCHY:
                seg_logits = cup_preserving_od_oc_logits(
                    raw_seg_logits,
                    detach_cup_from_od=self.detach_cup_from_od,
                )
        if self.segmentation_hierarchy == CONDITIONAL_CUP_HIERARCHY:
            seg_logits = hierarchy_constrained_od_oc_logits(seg_logits)
        elif self.segmentation_hierarchy == CUP_PRESERVING_HIERARCHY and not self.vertical_geometry_calibration_enabled:
            seg_logits = cup_preserving_od_oc_logits(
                seg_logits,
                detach_cup_from_od=self.detach_cup_from_od,
            )
        diagnostic_feature = fused.mean(dim=(-2, -1))
        if self.region_conditioned_classifier:
            # Use the predicted anatomical regions as soft attention pools.
            # This creates supervised cross-task coupling during source
            # training; it is deliberately separate from the failed
            # label-free C3-TTA objective.
            mask_prob = torch.sigmoid(seg_logits)
            mask_prob = torch.nn.functional.interpolate(
                mask_prob,
                size=fused.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            # The detached variant is a causal control: the classifier still
            # receives identical OD/OC soft regions, but its loss cannot
            # update segmentation logits through the pooling weights.
            if self.detach_region_masks:
                mask_prob = mask_prob.detach()
            pooled_regions = []
            for channel in range(2):
                weight = mask_prob[:, channel : channel + 1]
                pooled_regions.append(
                    (fused * weight).sum(dim=(-2, -1))
                    / weight.sum(dim=(-2, -1)).clamp_min(1e-4)
                )
            diagnostic_feature = torch.cat(
                [diagnostic_feature, pooled_regions[0], pooled_regions[1]], dim=1
            )
        cls_logits = self.classifier(diagnostic_feature)
        if self.global_classifier is not None:
            cls_logits = cls_logits + self.global_classifier(backbone_out.global_feat)
        return model_output_from_logits(
            seg_logits,
            cls_logits,
            adapted,
            boundary_logits=boundary_logits,
            anatomy_field_logits=anatomy_field_logits,
            vertical_geometry_logits=vertical_geometry_logits,
        )

    def freeze_backbone_and_heads(self) -> None:
        """Freeze every source-trained weight except explicit TTA adapters.

        Both decoder families implement ``tta_adapter_modules``.  The
        adapter-fusion decoder exposes one adapter per feature level; the
        DINOv3 Pyramid-FPN decoder exposes one adapter on its shared fused
        feature so segmentation and classification move together.
        """
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        adapters = self.tta_adapter_modules()
        if not adapters:
            raise RuntimeError(
                "This dense decoder has no test-time adapters. Use source-only inference "
                "or train a dedicated adaptation module before enabling TTA."
            )
        for adapter in adapters:
            for parameter in adapter.parameters():
                parameter.requires_grad_(True)

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        # LoRA is deliberately the only trainable portion of a frozen
        # foundation encoder.  A plain (non-LoRA) backbone returns an empty
        # list and therefore keeps the original source-training behavior.
        for parameter in getattr(self.backbone, "trainable_lora_parameters", lambda: [])():
            parameter.requires_grad_(True)

    def freeze_all_except_global_classifier(self) -> None:
        if self.global_classifier is None:
            raise RuntimeError(
                "global_residual_classifier must be enabled for global-classifier training"
            )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.global_classifier.parameters():
            parameter.requires_grad_(True)

    def enable_gradient_checkpointing(self) -> None:
        setter = getattr(self.backbone, "set_grad_checkpointing", None)
        if setter is not None:
            setter(True)

    def trainable_adapter_parameters(self):
        return [
            parameter
            for module in self.tta_adapter_modules()
            for parameter in module.parameters()
            if parameter.requires_grad
        ]

    def tta_adapter_modules(self) -> tuple[nn.Module, ...]:
        getter = getattr(self.decoder, "tta_adapter_modules", None)
        if getter is None:
            return ()
        return tuple(getter())

    def configure_tta_train_mode(self) -> None:
        """Enable gradients only through explicit TTA adapters in train mode."""
        self.eval()
        for module in self.tta_adapter_modules():
            module.train()

    def trainable_parameter_summary(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "trainable_percent": 100.0 * trainable / max(total, 1)}

    def freeze_state_snapshot(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in self.state_dict().items() if not self._is_trainable_name(name)}

    def _is_trainable_name(self, name: str) -> bool:
        return any(name == key or name.startswith(key + ".") for key, value in self.named_parameters() if value.requires_grad)
