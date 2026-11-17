"""Causal temporal convolutional network for observable shelf-interaction actions.

The model maps a feature sequence ``(B, C, T)`` to per-step action logits ``(B, K, T)``. Every
convolution is left-padded only, so the output at step ``t`` never depends on steps after ``t``.
That is the property live inference needs, and ``tests/test_tcn.py`` checks it.

Outputs are uncalibrated logits for *observable actions* (for example "reach to shelf"). They are
never a theft, intent or guilt score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class TCNConfig:
    in_channels: int
    classes: tuple[str, ...]
    hidden: int = 64
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.2
    temporal: bool = True  # False -> per-step linear baseline (no context)

    @property
    def receptive_field(self) -> int:
        if not self.temporal:
            return 1
        return 1 + sum(2 * (self.kernel_size - 1) * d for d in self.dilations)


class CausalConv1d(nn.Conv1d):
    def __init__(self, cin: int, cout: int, kernel_size: int, dilation: int) -> None:
        super().__init__(cin, cout, kernel_size, dilation=dilation)
        self._left = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return super().forward(nn.functional.pad(x, (self._left, 0)))


class _Block(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, kernel_size, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel_size, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return x + self.net(x)


class ActionTCN(nn.Module):
    def __init__(self, config: TCNConfig) -> None:
        super().__init__()
        self.config = config
        k = len(config.classes)
        if config.temporal:
            self.inp = nn.Sequential(nn.Dropout(config.dropout), nn.Conv1d(config.in_channels, config.hidden, 1), nn.GELU())
            self.body = nn.Sequential(*(_Block(config.hidden, config.kernel_size, d, config.dropout) for d in config.dilations))
            self.head = nn.Conv1d(config.hidden, k, 1)
        else:
            self.inp = nn.Dropout(config.dropout)
            self.body = nn.Identity()
            self.head = nn.Conv1d(config.in_channels, k, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, C, T) normalised features -> (B, K, T) logits."""
        return self.head(self.body(self.inp(x)))


@dataclass
class Checkpoint:
    config: TCNConfig
    state_dict: dict[str, torch.Tensor]
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    thresholds: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        torch.save(
            {
                "format": "shop_action_tcn_v1",
                "config": {**asdict(self.config), "classes": list(self.config.classes), "dilations": list(self.config.dilations)},
                "state_dict": self.state_dict,
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
                "thresholds": self.thresholds,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "Checkpoint":
        raw = torch.load(path, map_location="cpu", weights_only=True)
        if raw.get("format") != "shop_action_tcn_v1":
            raise ValueError("not a shop_action_tcn_v1 checkpoint")
        cfg = dict(raw["config"])
        cfg["classes"] = tuple(cfg["classes"])
        cfg["dilations"] = tuple(cfg["dilations"])
        return cls(TCNConfig(**cfg), raw["state_dict"], raw["feature_mean"], raw["feature_std"], list(raw["thresholds"]), dict(raw["metadata"]))

    def build(self) -> ActionTCN:
        model = ActionTCN(self.config)
        model.load_state_dict(self.state_dict)
        return model.eval()
