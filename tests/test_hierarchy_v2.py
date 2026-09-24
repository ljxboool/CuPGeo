from __future__ import annotations

from pathlib import Path

import pytest
import torch

from c3tta.losses.hierarchy import soft_topology_config, thresholded_topology_loss
from c3tta.losses.multitask import source_loss
from c3tta.models.backbones import build_backbone
from c3tta.models.config import Config, load_config
from c3tta.models.factory import (
    PYRAMID_FPN_LORA_GLOBAL_CLASSIFIER_MODEL_ARCHITECTURE,
    PYRAMID_FPN_LORA_MODEL_ARCHITECTURE,
    PYRAMID_FPN_LORA_CUP_PRESERVING_DAFL_MODEL_ARCHITECTURE,
    PYRAMID_FPN_LORA_CUP_PRESERVING_NO_DETACH_DAFL_MODEL_ARCHITECTURE,
    PYRAMID_FPN_LORA_GEOMETRY_COUPLED_DAFL_MODEL_ARCHITECTURE,
    build_model,
    model_architecture_for_config,
    segmentation_hierarchy_for_config,
    validate_checkpoint_architecture,
    validate_init_checkpoint_architecture,
)
from c3tta.models.multitask import (
    CONDITIONAL_CUP_HIERARCHY,
    CUP_PRESERVING_HIERARCHY,
    INDEPENDENT_SEGMENTATION,
    C3MultiTask,
    cup_preserving_od_oc_logits,
    hierarchy_constrained_od_oc_logits,
    resolve_segmentation_hierarchy_mode,
)
from scripts.train_source import delayed_auxiliary_loss_ramp


def _logit(probability: torch.Tensor) -> torch.Tensor:
    return torch.logit(probability.clamp(1e-5, 1.0 - 1e-5))


def test_legacy_conditional_cup_parameterization_is_unchanged() -> None:
    raw = torch.randn(2, 2, 7, 9)
    raw_float = raw.float()
    od = torch.sigmoid(raw_float[:, 0:1])
    oc = od * torch.sigmoid(raw_float[:, 1:2])
    expected = _logit(torch.cat((od, oc), dim=1))
    assert torch.equal(hierarchy_constrained_od_oc_logits(raw), expected)


def test_cup_preserving_parameterization_is_strict_and_keeps_oc_marginal() -> None:
    raw = torch.randn(2, 2, 7, 9, requires_grad=True)
    logits = cup_preserving_od_oc_logits(raw)
    probability = torch.sigmoid(logits)
    expected_oc = torch.sigmoid(raw[:, 1:2].float())
    assert torch.all(probability[:, 1:2] <= probability[:, 0:1])
    assert torch.allclose(probability[:, 1:2], expected_oc, atol=1e-6, rtol=1e-6)

    probability[:, 1:2].sum().backward()
    assert raw.grad is not None
    assert torch.count_nonzero(raw.grad[:, 0:1]) == 0
    assert torch.count_nonzero(raw.grad[:, 1:2]) > 0


def test_cup_preserving_od_branch_does_not_update_cup_logits() -> None:
    raw = torch.randn(2, 2, 5, 7, requires_grad=True)
    logits = cup_preserving_od_oc_logits(raw)
    logits[:, 0:1].sum().backward()
    assert raw.grad is not None
    assert torch.count_nonzero(raw.grad[:, 0:1]) > 0
    assert torch.count_nonzero(raw.grad[:, 1:2]) == 0
    detached_rim_gradient = raw.grad[:, 0:1].clone()

    coupled_raw = raw.detach().clone().requires_grad_()
    coupled_logits = cup_preserving_od_oc_logits(
        coupled_raw,
        detach_cup_from_od=False,
    )
    assert torch.equal(logits.detach(), coupled_logits.detach())
    coupled_logits[:, 0:1].sum().backward()
    assert coupled_raw.grad is not None
    assert torch.count_nonzero(coupled_raw.grad[:, 1:2]) > 0
    assert torch.equal(coupled_raw.grad[:, 0:1], detached_rim_gradient)


def test_cup_preserving_model_wires_detach_switch_without_forward_change() -> None:
    common = dict(
        backbone_dim=64,
        levels=4,
        decoder_channels=16,
        adapter_rank=4,
        decoder_type="pyramid_fpn",
        decoder_dropout=0.0,
        anatomy_field_head=True,
        hierarchical_segmentation=True,
        segmentation_hierarchy=CUP_PRESERVING_HIERARCHY,
    )
    detached = C3MultiTask(
        build_backbone("tiny"),
        detach_cup_from_od=True,
        **common,
    )
    coupled = C3MultiTask(
        build_backbone("tiny"),
        detach_cup_from_od=False,
        **common,
    )
    coupled.load_state_dict(detached.state_dict(), strict=True)
    detached.eval()
    coupled.eval()
    images = torch.randn(2, 3, 64, 64)

    detached_output = detached(images)
    coupled_output = coupled(images)
    assert torch.equal(detached_output.seg_logits, coupled_output.seg_logits)

    detached_output.seg_logits[:, 0:1].sum().backward()
    coupled_output.seg_logits[:, 0:1].sum().backward()
    detached_bias_grad = detached.decoder.segmentation_head.bias.grad
    coupled_bias_grad = coupled.decoder.segmentation_head.bias.grad
    assert detached_bias_grad is not None
    assert coupled_bias_grad is not None
    assert detached_bias_grad[1].item() == 0.0
    assert coupled_bias_grad[1].abs().item() > 0.0


def test_cup_preserving_parameterization_represents_nested_probabilities() -> None:
    target_od = torch.tensor([[[[0.30, 0.75, 0.95]]]])
    target_oc = torch.tensor([[[[0.10, 0.50, 0.90]]]])
    conditional_rim = (target_od - target_oc) / (1.0 - target_oc)
    raw = torch.cat((_logit(conditional_rim), _logit(target_oc)), dim=1)
    probability = torch.sigmoid(cup_preserving_od_oc_logits(raw))
    expected = torch.cat((target_od, target_oc), dim=1)
    assert torch.allclose(probability, expected, atol=1e-6, rtol=1e-6)


def test_cup_preserving_logit_formula_matches_asymmetric_channel_definition() -> None:
    raw = torch.tensor(
        [[[[-2.0, 1.0]], [[1.5, -0.5]]]],
        dtype=torch.float32,
    )
    rim = torch.sigmoid(raw[:, 0:1])
    cup = torch.sigmoid(raw[:, 1:2])
    expected = torch.cat((cup + (1.0 - cup) * rim, cup), dim=1)
    actual = torch.sigmoid(cup_preserving_od_oc_logits(raw))
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)


def test_cup_preserving_oc_gradient_is_not_gated_by_uncertain_od() -> None:
    raw_legacy = torch.tensor([[[[-1.3862944]], [[0.0]]]], requires_grad=True)
    legacy_oc = torch.sigmoid(hierarchy_constrained_od_oc_logits(raw_legacy))[:, 1]
    legacy_oc.sum().backward()
    assert raw_legacy.grad is not None

    raw_v2 = raw_legacy.detach().clone().requires_grad_()
    v2_oc = torch.sigmoid(cup_preserving_od_oc_logits(raw_v2))[:, 1]
    v2_oc.sum().backward()
    assert raw_v2.grad is not None
    assert raw_v2.grad[0, 1, 0, 0].abs() > raw_legacy.grad[0, 1, 0, 0].abs()


def test_cup_preserving_parameterization_is_finite_for_fp16_extremes() -> None:
    raw = torch.tensor(
        [[[[50.0, -50.0]], [[-50.0, 50.0]]]],
        dtype=torch.float16,
        requires_grad=True,
    )
    logits = cup_preserving_od_oc_logits(raw)
    probability = torch.sigmoid(logits)
    assert torch.isfinite(logits).all()
    assert torch.all(probability[:, 1] <= probability[:, 0])
    logits.square().mean().backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


def test_explicit_hierarchy_modes_preserve_legacy_config_compatibility() -> None:
    legacy = Config(
        model={
            "decoder_type": "pyramid_fpn",
            "backbone": "tiny",
            "lora_rank": 8,
            "anatomy_field_head": True,
            "hierarchical_segmentation": True,
        }
    )
    assert segmentation_hierarchy_for_config(legacy) == CONDITIONAL_CUP_HIERARCHY
    assert (
        model_architecture_for_config(legacy)
        == PYRAMID_FPN_LORA_GEOMETRY_COUPLED_DAFL_MODEL_ARCHITECTURE
    )

    cup_preserving = Config(
        model={
            **legacy.model,
            "segmentation_hierarchy": CUP_PRESERVING_HIERARCHY,
        }
    )
    assert segmentation_hierarchy_for_config(cup_preserving) == CUP_PRESERVING_HIERARCHY
    assert (
        model_architecture_for_config(cup_preserving)
        == PYRAMID_FPN_LORA_CUP_PRESERVING_DAFL_MODEL_ARCHITECTURE
    )

    no_detach = Config(
        model={
            **cup_preserving.model,
            "detach_cup_from_od": False,
        }
    )
    assert (
        model_architecture_for_config(no_detach)
        == PYRAMID_FPN_LORA_CUP_PRESERVING_NO_DETACH_DAFL_MODEL_ARCHITECTURE
    )
    no_detach_tiny = Config(
        model={
            "decoder_type": "pyramid_fpn",
            "backbone": "tiny",
            "lora_rank": 0,
            "anatomy_field_head": True,
            "segmentation_hierarchy": CUP_PRESERVING_HIERARCHY,
            "detach_cup_from_od": False,
        }
    )
    assert (
        build_model(no_detach_tiny, initialize_backbone=False).detach_cup_from_od
        is False
    )

    with pytest.raises(ValueError, match="requires.*cup_preserving"):
        model_architecture_for_config(
            Config(
                model={
                    "decoder_type": "pyramid_fpn",
                    "backbone": "tiny",
                    "lora_rank": 8,
                    "detach_cup_from_od": False,
                }
            )
        )

    with pytest.raises(ValueError, match="conflicts"):
        segmentation_hierarchy_for_config(
            Config(
                model={
                    "hierarchical_segmentation": False,
                    "segmentation_hierarchy": CUP_PRESERVING_HIERARCHY,
                }
            )
        )


def test_hierarchy_mode_resolution_distinguishes_missing_legacy_field() -> None:
    expected = (
        (None, None, INDEPENDENT_SEGMENTATION),
        (None, False, INDEPENDENT_SEGMENTATION),
        (None, True, CONDITIONAL_CUP_HIERARCHY),
        (INDEPENDENT_SEGMENTATION, None, INDEPENDENT_SEGMENTATION),
        (CONDITIONAL_CUP_HIERARCHY, None, CONDITIONAL_CUP_HIERARCHY),
        (CUP_PRESERVING_HIERARCHY, None, CUP_PRESERVING_HIERARCHY),
        (INDEPENDENT_SEGMENTATION, False, INDEPENDENT_SEGMENTATION),
        (CONDITIONAL_CUP_HIERARCHY, True, CONDITIONAL_CUP_HIERARCHY),
        (CUP_PRESERVING_HIERARCHY, True, CUP_PRESERVING_HIERARCHY),
    )
    for explicit, legacy, resolved in expected:
        assert (
            resolve_segmentation_hierarchy_mode(
                explicit,
                hierarchical_segmentation=legacy,
            )
            == resolved
        )

    for explicit, legacy in (
        (INDEPENDENT_SEGMENTATION, True),
        (CONDITIONAL_CUP_HIERARCHY, False),
        (CUP_PRESERVING_HIERARCHY, False),
    ):
        with pytest.raises(ValueError, match="conflicts"):
            resolve_segmentation_hierarchy_mode(
                explicit,
                hierarchical_segmentation=legacy,
            )


def test_hierarchy_modes_have_strict_state_dict_compatibility() -> None:
    independent = C3MultiTask(
        build_backbone("tiny"),
        backbone_dim=64,
        levels=4,
        decoder_channels=16,
        adapter_rank=4,
        decoder_type="pyramid_fpn",
        anatomy_field_head=True,
        hierarchical_segmentation=False,
        segmentation_hierarchy=INDEPENDENT_SEGMENTATION,
    )
    cup_preserving = C3MultiTask(
        build_backbone("tiny"),
        backbone_dim=64,
        levels=4,
        decoder_channels=16,
        adapter_rank=4,
        decoder_type="pyramid_fpn",
        anatomy_field_head=True,
        hierarchical_segmentation=True,
        segmentation_hierarchy=CUP_PRESERVING_HIERARCHY,
    )
    cup_preserving.load_state_dict(independent.state_dict(), strict=True)
    assert independent.state_dict().keys() == cup_preserving.state_dict().keys()


def test_cup_preserving_head_bias_matches_legacy_neutral_marginals() -> None:
    model = C3MultiTask(
        build_backbone("tiny"),
        backbone_dim=64,
        levels=4,
        decoder_channels=16,
        adapter_rank=4,
        decoder_type="pyramid_fpn",
        anatomy_field_head=True,
        segmentation_hierarchy=CUP_PRESERVING_HIERARCHY,
    )
    raw_bias = model.decoder.segmentation_head.bias.view(1, 2, 1, 1)
    probabilities = torch.sigmoid(cup_preserving_od_oc_logits(raw_bias))
    assert probabilities[0, 0, 0, 0].item() == pytest.approx(0.50, abs=1e-7)
    assert probabilities[0, 1, 0, 0].item() == pytest.approx(0.25, abs=1e-7)


def test_checkpoint_markers_reject_cross_hierarchy_initialization() -> None:
    cup_preserving = Config(
        model={
            "decoder_type": "pyramid_fpn",
            "backbone": "tiny",
            "lora_rank": 8,
            "anatomy_field_head": True,
            "segmentation_hierarchy": CUP_PRESERVING_HIERARCHY,
        }
    )
    legacy_checkpoint = {
        "model_architecture": (
            PYRAMID_FPN_LORA_GEOMETRY_COUPLED_DAFL_MODEL_ARCHITECTURE
        )
    }
    with pytest.raises(RuntimeError, match="does not match"):
        validate_checkpoint_architecture(legacy_checkpoint, cup_preserving)
    with pytest.raises(RuntimeError, match="cross-hierarchy"):
        validate_init_checkpoint_architecture(legacy_checkpoint, cup_preserving)

    global_classifier = Config(
        model={
            "decoder_type": "pyramid_fpn",
            "backbone": "tiny",
            "lora_rank": 8,
            "global_residual_classifier": True,
        }
    )
    assert (
        model_architecture_for_config(global_classifier)
        == PYRAMID_FPN_LORA_GLOBAL_CLASSIFIER_MODEL_ARCHITECTURE
    )
    assert validate_init_checkpoint_architecture(
        {"model_architecture": PYRAMID_FPN_LORA_MODEL_ARCHITECTURE},
        global_classifier,
    ) == ("global_classifier.",)
    assert validate_init_checkpoint_architecture(
        {
            "model_architecture": (
                PYRAMID_FPN_LORA_GLOBAL_CLASSIFIER_MODEL_ARCHITECTURE
            )
        },
        global_classifier,
    ) == ()


def test_thresholded_topology_ignores_legal_high_vcdr_probabilities() -> None:
    # The second pixel has p_OC > p_OD, but both are foreground at threshold
    # 0.5, so the evaluated binary topology is still legal.
    probability = torch.tensor([[[[0.80, 0.60]], [[0.79, 0.80]]]])
    logits = _logit(probability).requires_grad_()
    loss, stats = thresholded_topology_loss(
        logits,
        threshold=0.5,
        topk_fraction=1.0,
        gradient_target="disc_only",
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-8)
    assert stats["violating_image_fraction"] == 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0


def test_disc_only_soft_topology_repairs_od_without_shrinking_oc() -> None:
    probability = torch.tensor([[[[0.40]], [[0.60]]]])
    logits = _logit(probability).requires_grad_()
    loss, stats = thresholded_topology_loss(
        logits,
        threshold=0.5,
        topk_fraction=1.0,
        gradient_target="disc_only",
    )
    assert loss.item() > 0.0
    assert stats["violating_pixel_fraction"] == 1.0
    assert stats["violating_image_fraction"] == 1.0
    assert stats["active_topk_fraction"] == 1.0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 0, 0, 0] < 0.0
    assert logits.grad[0, 1, 0, 0] == 0.0


def test_soft_topology_reports_sparse_topk_dilution_explicitly() -> None:
    probability = torch.tensor(
        [[[[0.40, 0.80, 0.80, 0.80]], [[0.60, 0.20, 0.20, 0.20]]]]
    )
    loss, stats = thresholded_topology_loss(
        _logit(probability),
        threshold=0.5,
        topk_fraction=1.0,
        gradient_target="disc_only",
    )
    assert loss.item() == pytest.approx(0.05, abs=1e-7)
    assert stats["mean_violating_probability_excess"] == pytest.approx(
        0.20,
        abs=1e-7,
    )
    assert stats["active_topk_fraction"] == pytest.approx(0.25, abs=1e-7)


def test_symmetric_soft_topology_has_correct_gradient_directions() -> None:
    probability = torch.tensor([[[[0.40]], [[0.60]]]])
    logits = _logit(probability).requires_grad_()
    loss, _ = thresholded_topology_loss(
        logits,
        threshold=0.5,
        topk_fraction=1.0,
        gradient_target="both",
    )
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 0, 0, 0] < 0.0
    assert logits.grad[0, 1, 0, 0] > 0.0


def test_zero_soft_topology_weight_is_gradient_equivalent_to_no_hierarchy() -> None:
    seg_target = torch.randint(0, 2, (2, 2, 5, 7)).float()
    cls_target = torch.randint(0, 2, (2, 1)).float()
    base_seg = torch.randn(2, 2, 5, 7, requires_grad=True)
    base_cls = torch.randn(2, 1, requires_grad=True)
    base_loss, _ = source_loss(base_seg, base_cls, seg_target, cls_target)
    base_loss.backward()

    soft_seg = base_seg.detach().clone().requires_grad_()
    soft_cls = base_cls.detach().clone().requires_grad_()
    supervised, _ = source_loss(soft_seg, soft_cls, seg_target, cls_target)
    topology, _ = thresholded_topology_loss(
        soft_seg,
        threshold=0.5,
        topk_fraction=0.01,
        gradient_target="disc_only",
    )
    combined = supervised + 0.0 * topology
    combined.backward()

    assert torch.equal(base_loss.detach(), combined.detach())
    assert torch.equal(base_seg.grad, soft_seg.grad)
    assert torch.equal(base_cls.grad, soft_cls.grad)


def test_soft_topology_config_and_delayed_ramp_are_explicit() -> None:
    config = soft_topology_config(
        {
            "soft_topology": {
                "enabled": True,
                "weight": 0.05,
                "start_epoch": 12,
                "ramp_epochs": 12,
                "topk_fraction": 0.01,
                "gradient_target": "disc_only",
            }
        }
    )
    assert config["start_epoch"] == 12
    assert config["ramp_epochs"] == 12
    assert delayed_auxiliary_loss_ramp(11, start_epoch=12, ramp_epochs=12) == 0.0
    assert delayed_auxiliary_loss_ramp(12, start_epoch=12, ramp_epochs=12) == 0.0
    assert delayed_auxiliary_loss_ramp(18, start_epoch=12, ramp_epochs=12) == 0.5
    assert delayed_auxiliary_loss_ramp(24, start_epoch=12, ramp_epochs=12) == 1.0

    with pytest.raises(ValueError, match="topk_fraction"):
        soft_topology_config({"soft_topology": {"topk_fraction": 0.0}})
    with pytest.raises(ValueError, match="gradient_target"):
        soft_topology_config({"soft_topology": {"gradient_target": "cup_only"}})
    with pytest.raises(ValueError, match="threshold"):
        soft_topology_config({"soft_topology": {"threshold": 1.0}})
    with pytest.raises(ValueError, match="ramp_epochs"):
        soft_topology_config({"soft_topology": {"ramp_epochs": -1}})


def test_new_experiment_configs_resolve_only_the_intended_hierarchy() -> None:
    config_root = Path(__file__).resolve().parents[1] / "configs"
    cup_preserving = load_config(
        config_root
        / "cp_baseline.yaml"
    )
    soft = load_config(
        config_root
        / "soft_topology.yaml"
    )
    assert segmentation_hierarchy_for_config(cup_preserving) == CUP_PRESERVING_HIERARCHY
    assert soft.model["hierarchical_segmentation"] is False
    assert segmentation_hierarchy_for_config(soft) == INDEPENDENT_SEGMENTATION
    assert soft.train["soft_topology"]["enabled"] is True
    assert (
        cup_preserving.train["scale_normalized_anatomy_field"]
        == soft.train["scale_normalized_anatomy_field"]
    )
