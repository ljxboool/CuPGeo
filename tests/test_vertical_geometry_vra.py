from __future__ import annotations

import torch

from c3tta.data.dataset import vertical_geometry_target
from c3tta.losses.multitask import vertical_geometry_loss
from c3tta.models.config import Config
from c3tta.models.backbones import build_backbone
from c3tta.models.factory import (
    PYRAMID_FPN_LORA_CUP_PRESERVING_DAFL_MODEL_ARCHITECTURE,
    PYRAMID_FPN_LORA_CUP_PRESERVING_VRA_MODEL_ARCHITECTURE,
    model_architecture_for_config,
    validate_init_checkpoint_architecture,
)
from c3tta.models.multitask import (
    CUP_PRESERVING_HIERARCHY,
    cup_preserving_od_oc_logits,
    vertical_geometry_calibration,
    C3MultiTask,
)


def test_vertical_geometry_target_uses_pixel_edges_and_sums_to_one() -> None:
    target = torch.zeros(2, 10, 4)
    target[0, 2:9] = 1.0
    target[1, 4:7] = 1.0
    lengths = vertical_geometry_target(target)
    expected = torch.tensor([[0.2, 0.2, 0.3, 0.2, 0.1]])
    assert torch.allclose(lengths, expected, atol=1e-6)
    assert torch.allclose(lengths.sum(dim=1), torch.ones(1))


def test_vertical_geometry_loss_is_finite_and_backpropagates() -> None:
    logits = torch.randn(3, 5, requires_grad=True)
    target = torch.softmax(torch.randn(3, 5), dim=1)
    loss = vertical_geometry_loss(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_vertical_geometry_calibration_preserves_valid_output() -> None:
    raw = torch.randn(2, 2, 9, 11, requires_grad=True)
    geometry = torch.randn(2, 5, requires_grad=True)
    calibrated = vertical_geometry_calibration(
        raw,
        geometry,
        CUP_PRESERVING_HIERARCHY,
        alpha=0.4,
    )
    output = cup_preserving_od_oc_logits(calibrated)
    probability = torch.sigmoid(output)
    assert torch.isfinite(output).all()
    assert torch.all(probability[:, 1] <= probability[:, 0])
    output.square().mean().backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert geometry.grad is not None and torch.isfinite(geometry.grad).all()


def test_vertical_geometry_calibration_zero_alpha_is_identity() -> None:
    raw = torch.randn(1, 2, 5, 7)
    geometry = torch.randn(1, 5)
    calibrated = vertical_geometry_calibration(
        raw, geometry, CUP_PRESERVING_HIERARCHY, alpha=0.0
    )
    assert torch.equal(calibrated, raw)


def test_vra_checkpoint_expansion_is_explicitly_allowed() -> None:
    cfg = Config(
        model={
            "decoder_type": "pyramid_fpn",
            "backbone": "tiny",
            "lora_rank": 8,
            "anatomy_field_head": True,
            "hierarchical_segmentation": True,
            "segmentation_hierarchy": CUP_PRESERVING_HIERARCHY,
            "vertical_geometry_head": True,
        }
    )
    assert model_architecture_for_config(cfg) == PYRAMID_FPN_LORA_CUP_PRESERVING_VRA_MODEL_ARCHITECTURE
    checkpoint = {
        "model_architecture": PYRAMID_FPN_LORA_CUP_PRESERVING_DAFL_MODEL_ARCHITECTURE,
    }
    assert validate_init_checkpoint_architecture(checkpoint, cfg) == (
        "decoder.vertical_geometry_head.",
        "vertical_geometry_alpha",
    )


def test_vra_model_forward_exposes_geometry_logits_and_nested_masks() -> None:
    model = C3MultiTask(
        build_backbone("tiny"),
        backbone_dim=64,
        levels=4,
        decoder_channels=16,
        adapter_rank=4,
        decoder_type="pyramid_fpn",
        decoder_dropout=0.0,
        anatomy_field_head=True,
        hierarchical_segmentation=True,
        segmentation_hierarchy=CUP_PRESERVING_HIERARCHY,
        vertical_geometry_head=True,
        vertical_geometry_calibration_enabled=True,
    )
    output = model(torch.randn(2, 3, 64, 64))
    assert output.vertical_geometry_logits is not None
    assert output.vertical_geometry_logits.shape == (2, 5)
    probability = torch.sigmoid(output.seg_logits)
    assert torch.all(probability[:, 1] <= probability[:, 0])
