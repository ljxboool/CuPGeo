"""Training-only, DDP-safe feature-statistics augmentations.

The production V100 profile uses one image per rank.  Off-the-shelf
MixStyle/DSU implementations estimate their style pool from a local batch and
would therefore degenerate to an identity mapping.  This module keeps their
published feature-statistics formulas while pooling *detached* style statistics
across DDP ranks.  It never gathers activations with gradients and is an exact
identity in evaluation mode.

The configured placement is the shared Pyramid-FPN fused feature, immediately
before the common segmentation/classification pathway.  It is intentionally a
single shared intervention rather than separate perturbations on task heads.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn


_SUPPORTED_METHODS = frozenset({"none", "mixstyle", "dsu"})
_ALLOWED_CONFIG_KEYS = frozenset(
    {
        "method",
        "p",
        "eps",
        "alpha",
        "factor",
        "global_pool",
        "donor_policy",
        "placement",
    }
)


@dataclass(frozen=True)
class StyleAugmentationConfig:
    """Validated configuration for a source-only style baseline."""

    method: str = "none"
    p: float = 0.0
    eps: float = 1e-6
    alpha: float = 0.1
    factor: float = 1.0
    global_pool: bool = True
    donor_policy: str = "derangement"
    placement: str = "pyramid_fpn_fused"


def parse_style_augmentation_config(
    raw: Mapping[str, Any] | StyleAugmentationConfig | None,
) -> StyleAugmentationConfig:
    """Parse a strict, intentionally small baseline configuration surface."""
    if isinstance(raw, StyleAugmentationConfig):
        return raw
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("train.style_augmentation must be a mapping when provided")
    unknown = sorted(set(raw) - _ALLOWED_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Unsupported train.style_augmentation keys: {unknown}")

    method = str(raw.get("method", "none")).lower()
    if method == "maxstyle":
        raise ValueError(
            "MaxStyle is intentionally not exposed through this feature-statistics "
            "module: its official method requires an adversarial inner loop and an "
            "image-reconstruction branch. Use the separate MaxStyle-adapted protocol."
        )
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported style augmentation method {method!r}; "
            f"choose one of {sorted(_SUPPORTED_METHODS)}"
        )
    default_p = 0.0 if method == "none" else 0.5
    config = StyleAugmentationConfig(
        method=method,
        p=float(raw.get("p", default_p)),
        eps=float(raw.get("eps", 1e-6)),
        alpha=float(raw.get("alpha", 0.1)),
        factor=float(raw.get("factor", 1.0)),
        global_pool=bool(raw.get("global_pool", True)),
        donor_policy=str(raw.get("donor_policy", "derangement")).lower(),
        placement=str(raw.get("placement", "pyramid_fpn_fused")).lower(),
    )
    if not 0.0 <= config.p <= 1.0:
        raise ValueError("style augmentation p must be in [0, 1]")
    if config.eps <= 0.0:
        raise ValueError("style augmentation eps must be positive")
    if config.alpha <= 0.0:
        raise ValueError("MixStyle alpha must be positive")
    if config.factor < 0.0:
        raise ValueError("DSU factor must be non-negative")
    if config.placement != "pyramid_fpn_fused":
        raise ValueError(
            "Only the shared Pyramid-FPN fused-feature placement is supported; "
            f"got {config.placement!r}"
        )
    if config.method == "mixstyle" and config.donor_policy != "derangement":
        raise ValueError(
            "MixStyle uses the pre-registered cross-rank derangement donor policy"
        )
    return config


def style_method_from_config(
    raw: Mapping[str, Any] | StyleAugmentationConfig | None,
) -> str:
    """Return a validated method name for architecture/provenance checks."""
    return parse_style_augmentation_config(raw).method


class StyleAugmentation(nn.Module):
    """Published MixStyle/DSU statistics with a DDP-global source-style pool.

    A rank may retain gradients only through its own activation.  All donor
    statistics and uncertainty estimates are detached before collectives.  The
    random apply decision and all global random tensors are sampled by rank 0
    and broadcast, so every DDP process executes identical collectives.
    """

    def __init__(self, config: StyleAugmentationConfig) -> None:
        super().__init__()
        self.config = config

    def extra_repr(self) -> str:
        return (
            f"method={self.config.method!r}, p={self.config.p}, "
            f"global_pool={self.config.global_pool}, "
            f"placement={self.config.placement!r}"
        )

    def describe(self) -> dict[str, Any]:
        """Return checkpoint/log-friendly implementation metadata."""
        record = asdict(self.config)
        record.update(
            {
                "implementation": "ddp_global_feature_statistics_v1",
                "donor_statistics_detached": True,
                "evaluation_is_identity": True,
            }
        )
        return record

    @staticmethod
    def _distributed() -> bool:
        return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    @staticmethod
    def _validate_feature(x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                "Style augmentation expects an NCHW feature tensor after token-to-map "
                f"conversion, received shape {tuple(x.shape)}"
            )
        if not x.is_floating_point():
            raise ValueError("Style augmentation requires a floating-point feature tensor")

    def _sample_apply(self, x: torch.Tensor) -> bool:
        """Sample one synchronized Bernoulli decision for a training forward."""
        if self.config.p >= 1.0:
            return True
        if self.config.p <= 0.0:
            return False
        if not self._distributed():
            return bool(torch.rand((), device=x.device) < self.config.p)
        decision = torch.empty(1, dtype=torch.int64, device=x.device)
        if dist.get_rank() == 0:
            decision.fill_(int(torch.rand((), device=x.device) < self.config.p))
        dist.broadcast(decision, src=0)
        return bool(decision.item())

    def _shared_tensor(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        sampler: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        """Sample on rank 0 and broadcast if DDP is active."""
        if not self._distributed():
            return sampler()
        if dist.get_rank() == 0:
            # ``torch.distributions.Beta.sample`` may return a non-contiguous
            # CUDA view on older PyTorch/NCCL combinations. NCCL broadcast
            # rejects such views, so make the communication boundary explicit.
            result = sampler().contiguous()
            if tuple(result.shape) != shape or result.dtype != dtype or result.device != device:
                raise RuntimeError(
                    "Shared style sampler returned an unexpected tensor: "
                    f"shape={tuple(result.shape)}, dtype={result.dtype}, device={result.device}; "
                    f"expected shape={shape}, dtype={dtype}, device={device}"
                )
        else:
            result = torch.empty(shape, dtype=dtype, device=device)
        dist.broadcast(result, src=0)
        return result

    def _global_style_pool(
        self, local_statistics: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """Return detached global statistics and this rank's first index."""
        detached = local_statistics.detach().contiguous()
        if not (self.config.global_pool and self._distributed()):
            return detached, 0
        local_count = torch.tensor([detached.shape[0]], dtype=torch.int64, device=detached.device)
        counts = [torch.empty_like(local_count) for _ in range(dist.get_world_size())]
        dist.all_gather(counts, local_count)
        count_values = [int(value.item()) for value in counts]
        if len(set(count_values)) != 1:
            raise RuntimeError(
                "DDP style augmentation requires equal per-rank batch sizes; "
                f"received {count_values}. Keep train drop_last=True."
            )
        gathered = [torch.empty_like(detached) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, detached)
        rank = dist.get_rank()
        return torch.cat(gathered, dim=0), rank * detached.shape[0]

    def _donor_permutation(self, total: int, x: torch.Tensor) -> torch.Tensor:
        """Sample a random derangement so a batch-one rank always has a donor."""
        if total < 2:
            return torch.arange(total, device=x.device, dtype=torch.long)

        def sample() -> torch.Tensor:
            reference = torch.arange(total, device=x.device, dtype=torch.long)
            for _ in range(32):
                permutation = torch.randperm(total, device=x.device)
                if not torch.any(permutation == reference):
                    return permutation
            # This deterministic fallback is still a valid derangement and
            # prevents an adversarially unlucky RNG stream from looping.
            shift = int(torch.randint(1, total, (), device=x.device).item())
            return (reference + shift) % total

        return self._shared_tensor((total,), torch.long, x.device, sample)

    def _mixstyle(self, x: torch.Tensor) -> torch.Tensor:
        # Match the reference formula: normalise content using detached sample
        # style statistics, then interpolate with a donor's style.  FP32
        # statistics avoid half-precision variance instability on V100.
        work = x.float()
        mean = work.mean(dim=(2, 3))
        std = (work.var(dim=(2, 3)) + self.config.eps).sqrt()
        pool_mean, offset = self._global_style_pool(mean)
        pool_std, offset_std = self._global_style_pool(std)
        if offset != offset_std:
            raise AssertionError("style statistic pools disagree on local rank offset")
        total = int(pool_mean.shape[0])
        if total < 2:
            return x
        permutation = self._donor_permutation(total, x)

        def sample_lambda() -> torch.Tensor:
            concentration = torch.tensor(
                self.config.alpha, dtype=work.dtype, device=work.device
            )
            return torch.distributions.Beta(concentration, concentration).sample((total, 1))

        mixing = self._shared_tensor((total, 1), work.dtype, work.device, sample_lambda)
        local_indices = torch.arange(x.shape[0], device=x.device) + offset
        own_mean = pool_mean[local_indices]
        own_std = pool_std[local_indices]
        donor_mean = pool_mean[permutation[local_indices]]
        donor_std = pool_std[permutation[local_indices]]
        local_mixing = mixing[local_indices]
        mixed_mean = own_mean * (1.0 - local_mixing) + donor_mean * local_mixing
        mixed_std = own_std * (1.0 - local_mixing) + donor_std * local_mixing
        normalized = (work - mean.detach().unsqueeze(-1).unsqueeze(-1)) / std.detach().unsqueeze(
            -1
        ).unsqueeze(-1)
        result = normalized * mixed_std.unsqueeze(-1).unsqueeze(-1) + mixed_mean.unsqueeze(
            -1
        ).unsqueeze(-1)
        return result.to(dtype=x.dtype)

    def _dsu(self, x: torch.Tensor) -> torch.Tensor:
        # DSU uses the official sample mean/std reparameterization.  Global
        # uncertainty is estimated from detached DDP statistics; the current
        # rank's mean/std remain differentiable just as in the reference code.
        work = x.float()
        mean = work.mean(dim=(2, 3))
        std = (work.var(dim=(2, 3)) + self.config.eps).sqrt()
        pool_mean, offset = self._global_style_pool(mean)
        pool_std, offset_std = self._global_style_pool(std)
        if offset != offset_std:
            raise AssertionError("style statistic pools disagree on local rank offset")
        total = int(pool_mean.shape[0])
        if total < 2:
            return x
        uncertainty_mean = (pool_mean.var(dim=0) + self.config.eps).sqrt()
        uncertainty_std = (pool_std.var(dim=0) + self.config.eps).sqrt()

        def sample_noise() -> torch.Tensor:
            return torch.randn((2, total, mean.shape[1]), dtype=work.dtype, device=work.device)

        noise = self._shared_tensor(
            (2, total, mean.shape[1]), work.dtype, work.device, sample_noise
        )
        local_indices = torch.arange(x.shape[0], device=x.device) + offset
        beta = mean + self.config.factor * noise[0, local_indices] * uncertainty_mean
        gamma = (
            std + self.config.factor * noise[1, local_indices] * uncertainty_std
        ).clamp_min(self.config.eps)
        normalized = (work - mean.unsqueeze(-1).unsqueeze(-1)) / std.unsqueeze(-1).unsqueeze(
            -1
        )
        result = normalized * gamma.unsqueeze(-1).unsqueeze(-1) + beta.unsqueeze(-1).unsqueeze(
            -1
        )
        return result.to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_feature(x)
        if (
            not self.training
            or self.config.method == "none"
            or self.config.p <= 0.0
            or x.shape[0] == 0
            or x.shape[-2] * x.shape[-1] < 2
        ):
            return x
        if not self._sample_apply(x):
            return x
        if self.config.method == "mixstyle":
            return self._mixstyle(x)
        if self.config.method == "dsu":
            return self._dsu(x)
        raise AssertionError(f"missing implementation for {self.config.method!r}")


def build_style_augmentation(
    raw: Mapping[str, Any] | StyleAugmentationConfig | None,
) -> StyleAugmentation | None:
    """Build an optional module; absent/none preserves legacy model behavior."""
    config = parse_style_augmentation_config(raw)
    return None if config.method == "none" else StyleAugmentation(config)
