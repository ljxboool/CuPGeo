from __future__ import annotations

import pytest
import torch

from c3tta.losses.anatomy_field import (
    field_mask_consistency_loss,
    scale_normalized_anatomy_field_config,
)
from c3tta.losses.scale_equivariance import paired_scale_equivariance_config
from c3tta.models.backbones import build_backbone
from c3tta.models.config import Config
from c3tta.models.factory import (
    PYRAMID_FPN_LORA_GEOMETRY_COUPLED_DAFL_MODEL_ARCHITECTURE,
    model_architecture_for_config,
)
from c3tta.models.multitask import C3MultiTask, hierarchy_constrained_od_oc_logits
from scripts.train_source import auxiliary_loss_warmup


def test_hierarchical_logits_enforce_containment_and_backpropagate() -> None:
    raw_logits = torch.randn(2, 2, 9, 11, requires_grad=True)
    constrained = hierarchy_constrained_od_oc_logits(raw_logits)
    probabilities = torch.sigmoid(constrained)
    assert torch.all(probabilities[:, 1] <= probabilities[:, 0])
    constrained.square().mean().backward()
    assert raw_logits.grad is not None
    assert torch.isfinite(raw_logits.grad).all()


def test_geometry_coupled_model_enforces_containment() -> None:
    model = C3MultiTask(
        build_backbone("tiny"),
        backbone_dim=64,
        levels=4,
        decoder_channels=32,
        adapter_rank=8,
        decoder_type="pyramid_fpn",
        anatomy_field_head=True,
        hierarchical_segmentation=True,
    )
    output = model(torch.randn(2, 3, 64, 64))
    probability = torch.sigmoid(output.seg_logits)
    assert output.anatomy_field_logits is not None
    assert torch.all(probability[:, 1] <= probability[:, 0])


def test_field_mask_consistency_uses_the_field_sign_convention() -> None:
    field_logits = torch.tensor([[[[-1.0, 0.0, 1.0]], [[1.0, 0.0, -1.0]]]])
    temperature = 0.10
    aligned_segmentation_logits = field_logits / temperature
    opposite_segmentation_logits = -aligned_segmentation_logits
    aligned = field_mask_consistency_loss(
        aligned_segmentation_logits, field_logits, temperature=temperature
    )
    opposite = field_mask_consistency_loss(
        opposite_segmentation_logits, field_logits, temperature=temperature
    )
    assert aligned.item() == pytest.approx(0.0, abs=1e-7)
    assert opposite > aligned


def test_geometry_coupled_architecture_and_warmup_are_explicit() -> None:
    cfg = Config(
        model={
            "decoder_type": "pyramid_fpn",
            "backbone": "tiny",
            "lora_rank": 8,
            "anatomy_field_head": True,
            "hierarchical_segmentation": True,
        }
    )
    assert (
        model_architecture_for_config(cfg)
        == PYRAMID_FPN_LORA_GEOMETRY_COUPLED_DAFL_MODEL_ARCHITECTURE
    )
    assert auxiliary_loss_warmup(0, 12) == 0.0
    assert auxiliary_loss_warmup(6, 12) == 0.5
    assert auxiliary_loss_warmup(12, 12) == 1.0
    assert paired_scale_equivariance_config({"paired_scale_equivariance": {}})[
        "warmup_epochs"
    ] == 0
    assert scale_normalized_anatomy_field_config(
        {"scale_normalized_anatomy_field": {}}
    )["warmup_epochs"] == 0


def test_hierarchical_profile_requires_anatomy_field() -> None:
    with pytest.raises(ValueError, match="requires the anatomy-field"):
        model_architecture_for_config(
            Config(
                model={
                    "decoder_type": "pyramid_fpn",
                    "hierarchical_segmentation": True,
                }
            )
        )
