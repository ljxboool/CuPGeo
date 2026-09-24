from __future__ import annotations

import math

import torch
from torch import nn


class FeatureAdapter(nn.Module):
    """Residual bottleneck adapter; zero initialized for stable source transfer."""

    def __init__(
        self, channels: int, rank: int = 32, scale_init: float = 1e-2
    ) -> None:
        super().__init__()
        if not math.isfinite(scale_init) or scale_init <= 0.0:
            raise ValueError("FeatureAdapter scale_init must be finite and positive")
        self.norm = nn.GroupNorm(1, channels)
        self.down = nn.Conv2d(channels, rank, 1)
        self.act = nn.GELU()
        self.up = nn.Conv2d(rank, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.up(self.act(self.down(self.norm(x))))
