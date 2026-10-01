"""Synthetic shortcut injection for ID/OOD stress tests (Phase 1)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

__all__ = [
    "SyntheticShortcutInjector",
    "build_shortcut_injector",
    "correlation_with_target",
    "DEFAULT_SINE_HUM_CONFIG",
]

DEFAULT_SINE_HUM_CONFIG = "configs/shortcuts/sine_hum.yaml"
DEFAULT_BASELINE_DRIFT_CONFIG = "configs/shortcuts/baseline_drift.yaml"

# Canonical pilot defaults (mirrored in configs/shortcuts/sine_hum.yaml).
_DEFAULTS = {
    "shortcut_type": "sine_hum",
    "channel_idx": -1,
    "amplitude": 2.0,
    "freq": 0.1,
    "correlation_strength": 0.92,
    "ood_freq_multiplier": 3.0,
}


def correlation_with_target(shortcut: np.ndarray, target: np.ndarray) -> float:
    """Pearson correlation; returns 0.0 if either series has zero variance."""
    if shortcut.shape != target.shape:
        raise ValueError("shortcut and target must have the same shape")
    s = np.asarray(shortcut, dtype=np.float64).reshape(-1)
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    if s.size == 0:
        return 0.0
    if np.std(s) < 1e-12 or np.std(t) < 1e-12:
        return 0.0
    return float(np.corrcoef(s, t)[0, 1])


def build_shortcut_injector(
    config_path: str | None = None,
) -> SyntheticShortcutInjector:
    """Load the active shortcut injector from YAML (project-relative path)."""
    path = _resolve_config_path(config_path or DEFAULT_SINE_HUM_CONFIG)
    return SyntheticShortcutInjector.from_yaml(path)


class SyntheticShortcutInjector:
    """
    Target-aligned shortcut injector.
    Default channel_idx=-1 (OT / last channel) so channel-independent models
    (e.g. PatchTST) still see the spurious cue on the forecast target.
    """

    def __init__(
        self,
        shortcut_type: str | None = None,
        channel_idx: int | None = None,
        amplitude: float | None = None,
        freq: float | None = None,
        correlation_strength: float | None = None,
        ood_freq_multiplier: float | None = None,
    ):
        self.shortcut_type = shortcut_type or _DEFAULTS["shortcut_type"]
        self.channel_idx = (
            _DEFAULTS["channel_idx"] if channel_idx is None else channel_idx
        )
        self.amplitude = amplitude if amplitude is not None else _DEFAULTS["amplitude"]
        self.freq = freq if freq is not None else _DEFAULTS["freq"]
        self.correlation_strength = (
            correlation_strength
            if correlation_strength is not None
            else _DEFAULTS["correlation_strength"]
        )
        self.ood_freq_multiplier = (
            ood_freq_multiplier
            if ood_freq_multiplier is not None
            else _DEFAULTS["ood_freq_multiplier"]
        )
        # Index into ``y`` when building the target reference (OT = last column).
        self.target_idx = -1

    @classmethod
    def from_yaml(cls, config_path: str) -> SyntheticShortcutInjector:
        resolved = _resolve_config_path(config_path)
        path = Path(resolved)
        if not path.is_file():
            raise FileNotFoundError(f"Shortcut config not found: {config_path}")
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict) or "shortcut" not in raw:
            raise ValueError(f"Invalid shortcut YAML layout in {config_path}")
        return cls.from_config(raw["shortcut"])

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> SyntheticShortcutInjector:
        shortcut_type = cfg.get("type") or cfg.get("shortcut_type", _DEFAULTS["shortcut_type"])

        channel_idx = cfg.get("channel_idx")
        if channel_idx is None:
            channels = cfg.get("channels")
            channel_idx = int(channels[0]) if channels else _DEFAULTS["channel_idx"]
        else:
            channel_idx = int(channel_idx)

        train = cfg.get("train") or {}
        amplitude = cfg.get("amplitude", train.get("amplitude", _DEFAULTS["amplitude"]))

        freq = cfg.get("freq")
        if freq is None and train.get("omega") is not None:
            # Legacy: omega in rad/step -> freq in cycles/step for sin(2*pi*freq*t)
            freq = float(train["omega"]) / (2.0 * np.pi)
        if freq is None:
            freq = _DEFAULTS["freq"]

        correlation_strength = cfg.get(
            "correlation_strength",
            train.get("target_corr_min", _DEFAULTS["correlation_strength"]),
        )

        ood = cfg.get("ood_test") or {}
        ood_freq_multiplier = float(
            ood.get("omega_multiplier", _DEFAULTS["ood_freq_multiplier"])
        )

        return cls(
            shortcut_type=str(shortcut_type),
            channel_idx=channel_idx,
            amplitude=float(amplitude),
            freq=float(freq),
            correlation_strength=float(correlation_strength),
            ood_freq_multiplier=ood_freq_multiplier,
        )

    def inject_shortcut(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        is_ood: bool = False,
    ) -> torch.Tensor:
        x_mod = x.clone()
        b, seq_len, _d = x_mod.shape
        device = x.device

        if y is not None:
            target_signal = (
                torch.mean(y[:, :, -1:], dim=1, keepdim=True).repeat(1, seq_len, 1).squeeze(-1)
            )
        else:
            target_signal = x_mod[:, :, -1]

        t = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(0).repeat(b, 1)

        if self.shortcut_type == "sine_hum":
            if not is_ood:
                phase_shift = target_signal * 0.5
                cue = self.amplitude * torch.sin(2.0 * np.pi * self.freq * t + phase_shift)
                shortcut = (self.correlation_strength * target_signal) + (
                    (1.0 - self.correlation_strength) * cue
                )
            else:
                shifted_freq = self.freq * self.ood_freq_multiplier
                cue = self.amplitude * torch.sin(
                    2.0 * np.pi * shifted_freq * t + (np.pi / 2.0)
                )
                shortcut = (-self.correlation_strength * target_signal) + cue

            x_mod[:, :, self.channel_idx] += shortcut

        elif self.shortcut_type == "baseline_drift":
            if not is_ood:
                drift = (self.amplitude / seq_len) * t * torch.sign(target_signal + 1e-4)
            else:
                drift = -(self.amplitude / seq_len) * t * torch.sign(target_signal + 1e-4)
            x_mod[:, :, self.channel_idx] += drift

        else:
            raise ValueError(f"Unknown shortcut_type: {self.shortcut_type}")

        return x_mod


def _resolve_config_path(config_path: str) -> str:
    """Resolve config relative to repo root when cwd differs (e.g. scripts/)."""
    if os.path.isfile(config_path):
        return config_path
    root = Path(__file__).resolve().parents[2]
    candidate = root / config_path
    if candidate.is_file():
        return str(candidate)
    return config_path
