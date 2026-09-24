from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .adapters import FeatureAdapter


class FeatureFusionDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        levels: int = 4,
        decoder_channels: int = 128,
        adapter_rank: int = 32,
        out_channels: int = 2,
        adapter_scale_init: float = 1e-2,
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, decoder_channels, 1) for _ in range(levels)])
        self.adapters = nn.ModuleList(
            [
                FeatureAdapter(
                    decoder_channels, adapter_rank, scale_init=adapter_scale_init
                )
                for _ in range(levels)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(decoder_channels, out_channels, 1),
        )

    def forward(
        self,
        features: list[torch.Tensor],
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor | None]:
        projected = []
        adapted = []
        for feature, projection, adapter in zip(features, self.projections, self.adapters):
            x = projection(feature)
            x = adapter(x)
            adapted.append(x)
            projected.append(F.interpolate(x, size=features[0].shape[-2:], mode="bilinear", align_corners=False))
        fused = torch.stack(projected, dim=0).mean(dim=0)
        logits = self.fuse(fused)
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits, adapted, fused

    def tta_adapter_modules(self) -> tuple[nn.Module, ...]:
        """Return the only decoder modules that online TTA may update."""
        return tuple(self.adapters)


def _group_norm(channels: int) -> nn.Module:
    """Return a GroupNorm that also works for batch size one.

    Source training on the 16 GB V100 profile intentionally uses one image per
    GPU.  BatchNorm would make the decoder statistics noisy, so the new dense
    decoder uses GroupNorm throughout.
    """
    # PPM includes a 1x1 branch.  With batch size one, one channel per group
    # would leave GroupNorm with a single value and PyTorch rightfully rejects
    # it in training mode.  Keeping at least two channels per group makes this
    # normalization valid for the V100 profile as well as larger batches.
    if channels == 1:
        return nn.Identity()
    for groups in range(min(32, channels // 2), 0, -1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    raise AssertionError("every positive channel count is divisible by one")


class _ConvNormGELU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            _group_norm(out_channels),
            nn.GELU(),
        )


class PyramidPoolingModule(nn.Module):
    """UPerNet-style pyramid pooling on the deepest ViT feature map."""

    def __init__(self, channels: int, bins: Sequence[int] = (1, 2, 3, 6)) -> None:
        super().__init__()
        if not bins or any(int(bin_size) <= 0 for bin_size in bins):
            raise ValueError("ppm bins must contain positive pooling sizes")
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(int(bin_size)),
                    nn.Conv2d(channels, channels, 1, bias=False),
                    _group_norm(channels),
                    nn.GELU(),
                )
                for bin_size in bins
            ]
        )
        self.bottleneck = _ConvNormGELU(channels * (len(self.stages) + 1), channels)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        target_size = feature.shape[-2:]
        pooled = [feature]
        for stage in self.stages:
            pooled.append(
                F.interpolate(
                    stage(feature),
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return self.bottleneck(torch.cat(pooled, dim=1))


class CBAM(nn.Module):
    """Convolutional Block Attention Module for fundus boundary refinement.

    The module follows the channel-then-spatial attention design used by
    boundary-aware medical segmentation adapters.  It is intentionally small:
    the frozen foundation encoder remains the source of semantic features,
    while CBAM learns where the OD/OC boundary evidence is reliable.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        if channels <= 0 or reduction <= 0:
            raise ValueError("CBAM channels and reduction must be positive")
        hidden = max(channels // reduction, 4)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        average = x.mean(dim=(2, 3), keepdim=True)
        maximum = x.amax(dim=(2, 3), keepdim=True)
        channel_gate = torch.sigmoid(self.channel_mlp(average) + self.channel_mlp(maximum))
        x = x * channel_gate
        spatial = torch.cat(
            [x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1
        )
        return x * torch.sigmoid(self.spatial(spatial))


class PyramidFPNDecoder(nn.Module):
    """A dense decoder for foundation-model ViT features.

    DINO features from multiple Transformer depths have a common token grid,
    unlike the native multi-resolution stages of a ConvNet.  This module first
    aligns them to one channel width, enriches the deepest semantic feature
    with pyramid pooling, then applies top-down FPN fusion across depths.  A
    lightweight full-resolution refinement head is deliberately narrow so the
    640 px V100 profile remains practical.
    """

    def __init__(
        self,
        in_channels: int,
        levels: int = 4,
        decoder_channels: int = 256,
        out_channels: int = 2,
        dropout: float = 0.1,
        ppm_bins: Sequence[int] = (1, 2, 3, 6),
        tta_adapter_rank: int = 0,
        tta_adapter_scale_init: float = 1e-2,
        boundary_aware: bool = False,
        anatomy_field_head: bool = False,
        anatomy_field_out_channels: int = 2,
        vertical_geometry_head: bool = False,
        style_augmentation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if levels < 2:
            raise ValueError("PyramidFPNDecoder needs at least two feature levels")
        if decoder_channels <= 0:
            raise ValueError("decoder_channels must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("decoder dropout must be in [0, 1)")
        if tta_adapter_rank < 0:
            raise ValueError("tta_adapter_rank must be non-negative")
        if anatomy_field_out_channels <= 0:
            raise ValueError("anatomy_field_out_channels must be positive")
        self.levels = levels
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(in_channels, decoder_channels, 1, bias=False) for _ in range(levels)]
        )
        self.ppm = PyramidPoolingModule(decoder_channels, ppm_bins)
        self.smooth_convs = nn.ModuleList(
            [_ConvNormGELU(decoder_channels, decoder_channels) for _ in range(levels)]
        )
        self.fuse = nn.Sequential(
            _ConvNormGELU(levels * decoder_channels, decoder_channels),
            nn.Dropout2d(dropout),
            _ConvNormGELU(decoder_channels, decoder_channels),
        )
        self.boundary_aware = bool(boundary_aware)
        self.anatomy_field_head_enabled = bool(anatomy_field_head)
        self.vertical_geometry_head_enabled = bool(vertical_geometry_head)
        self.fused_attention = CBAM(decoder_channels) if self.boundary_aware else nn.Identity()
        # Source-domain MixStyle/DSU baselines operate once on the shared FPN
        # feature, before both segmentation and classification consume it.
        # Identity carries no state, so legacy checkpoints remain strict-load
        # compatible when no augmentation is configured.
        self.style_augmentation = style_augmentation or nn.Identity()
        # This adapter is deliberately placed after multi-scale fusion and
        # before both task heads.  It is zero initialized, so expanding a
        # source-only DINOv3 Pyramid-FPN checkpoint preserves its initial
        # predictions exactly while exposing a compact shared feature for
        # C3-TTA updates at deployment time.
        self.tta_adapter: FeatureAdapter | None
        if tta_adapter_rank:
            self.tta_adapter = FeatureAdapter(
                decoder_channels,
                tta_adapter_rank,
                scale_init=tta_adapter_scale_init,
            )
        else:
            self.tta_adapter = None
        # Keep the feature map at the image resolution compact.  It supplies
        # local vessel/disc-boundary detail without storing a 256-channel 640²
        # activation on every V100 rank.
        refine_channels = min(64, decoder_channels)
        self.full_resolution_refine = nn.Sequential(
            _ConvNormGELU(decoder_channels, refine_channels),
            nn.Dropout2d(dropout),
            _ConvNormGELU(refine_channels, refine_channels),
        )
        self.segmentation_head = nn.Conv2d(refine_channels, out_channels, 1)
        self.anatomy_field_head: nn.Module | None
        if self.anatomy_field_head_enabled:
            # A separate continuous head avoids forcing the categorical mask
            # logits to serve as a physically meaningful distance coordinate.
            self.anatomy_field_head = nn.Sequential(
                _ConvNormGELU(refine_channels, refine_channels),
                nn.Conv2d(refine_channels, anatomy_field_out_channels, 1),
            )
        else:
            self.anatomy_field_head = None
        self.vertical_geometry_head: nn.Module | None
        if self.vertical_geometry_head_enabled:
            self.vertical_geometry_head = nn.Sequential(
                nn.LayerNorm(refine_channels),
                nn.Linear(refine_channels, 5),
            )
        else:
            self.vertical_geometry_head = None
        self.boundary_head: nn.Module | None
        if self.boundary_aware:
            self.boundary_head = nn.Sequential(
                _ConvNormGELU(refine_channels, refine_channels),
                nn.Conv2d(refine_channels, 1, 1),
            )
        else:
            self.boundary_head = None

    def forward(
        self,
        features: list[torch.Tensor],
        output_size: tuple[int, int],
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if len(features) != self.levels:
            raise ValueError(
                f"Expected {self.levels} Transformer features, received {len(features)}"
            )
        pyramid = [conv(feature) for conv, feature in zip(self.lateral_convs, features)]
        pyramid[-1] = self.ppm(pyramid[-1])
        for level in range(self.levels - 2, -1, -1):
            pyramid[level] = pyramid[level] + F.interpolate(
                pyramid[level + 1],
                size=pyramid[level].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        pyramid = [smooth(feature) for smooth, feature in zip(self.smooth_convs, pyramid)]
        target_size = pyramid[0].shape[-2:]
        fused = self.fuse(
            torch.cat(
                [
                    F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
                    for feature in pyramid
                ],
                dim=1,
            )
        )
        fused = self.fused_attention(fused)
        fused = self.style_augmentation(fused)
        if self.tta_adapter is not None:
            fused = self.tta_adapter(fused)
        full_resolution = F.interpolate(
            fused, size=output_size, mode="bilinear", align_corners=False
        )
        refined = self.full_resolution_refine(full_resolution)
        logits = self.segmentation_head(refined)
        anatomy_field_logits = (
            self.anatomy_field_head(refined)
            if self.anatomy_field_head is not None
            else None
        )
        boundary_logits = (
            self.boundary_head(refined) if self.boundary_head is not None else None
        )
        vertical_geometry_logits = (
            self.vertical_geometry_head(refined.mean(dim=(-2, -1)))
            if self.vertical_geometry_head is not None
            else None
        )
        return logits, pyramid, fused, boundary_logits, anatomy_field_logits, vertical_geometry_logits

    def tta_adapter_modules(self) -> tuple[nn.Module, ...]:
        """Expose the shared fused-feature adapter to the generic TTA engine."""
        return () if self.tta_adapter is None else (self.tta_adapter,)
