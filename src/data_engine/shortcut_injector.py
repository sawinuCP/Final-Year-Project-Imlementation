"""Synthetic shortcut injection engine (Phase 1).

Target-aligned spurious correlations injected into input covariate channels.

Type A -- Device-ID Sine Hum (``sine_hum``):
    In-distribution: shortcut blended with target (OT) reference (~92% alignment).
    OOD test split: inverted correlation, frequency x3, phase shift.

Type B -- Sensor-ID Linear Baseline Drift (``baseline_drift``):
    In-distribution: drift signed by target magnitude.
    OOD test split: inverted drift against target.

Injection applies to INPUT windows only ([B, L, D] FP32); targets stay clean unless
``y`` is passed only as a reference for correlation binding (not modified).

Verification: ``correlation_with_target`` + ``tests/test_shortcut_injector.py``.
Configs: ``configs/shortcuts/*.yaml`` via ``SyntheticShortcutInjector.from_yaml``.
"""

from __future__ import annotations

import numpy as np
import torch
import yaml

__all__ = ["SyntheticShortcutInjector", "correlation_with_target"]


class SyntheticShortcutInjector:
    """
    Target-Aligned Synthetic Shortcut Injector.

    In-Distribution (Train):
      Injects a shortcut strongly correlated (>= 90%) with the target variable Y (OT).
      The network learns to cheat by relying on this channel instead of true physics.

    Out-of-Distribution (Test Shift):
      Breaks or inverts the correlation (rho -> -rho or shifts frequency/phase).
      Causes standard ERM models to fail catastrophically.
    """

    def __init__(
        self,
        shortcut_type="sine_hum",
        channel_idx=0,
        target_idx=-1,
        amplitude=2.0,
        freq=0.1,
        correlation_strength=0.92,
    ):
        self.shortcut_type = shortcut_type
        self.channel_idx = channel_idx
        self.target_idx = target_idx
        self.amplitude = amplitude
        self.freq = freq
        self.correlation_strength = correlation_strength

    @classmethod
    def from_yaml(cls, yaml_path: str) -> SyntheticShortcutInjector:
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f).get("shortcut", {})

        train = raw.get("train", {})
        channels = raw.get("channels", [0])
        omega = train.get("omega")
        freq = train.get("freq")
        if freq is None and omega is not None:
            freq = float(omega) / (2.0 * np.pi)

        kwargs = {
            "shortcut_type": raw.get("type", raw.get("shortcut_type", "sine_hum")),
            "channel_idx": int(channels[0]) if channels else 0,
            "amplitude": train.get("amplitude", train.get("alpha", 2.0)),
            "freq": freq if freq is not None else 0.1,
            "correlation_strength": train.get(
                "target_corr_min", raw.get("correlation_strength", 0.92)
            ),
        }
        return cls(**kwargs)

    def inject_shortcut(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        is_ood: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: Input sequence [B, L, D]
            y: Target sequence [B, H, D] or [B, H, 1] (reference for alignment)
            is_ood: If True, breaks/inverts shortcut correlation.
        """
        x_mod = x.clone()
        b, l, _d = x_mod.shape
        device = x.device

        if y is not None:
            if y.shape[-1] == 1:
                target_signal = torch.mean(y, dim=1, keepdim=True)
            else:
                target_signal = torch.mean(
                    y[:, :, self.target_idx], dim=1, keepdim=True
                )
            target_signal = (
                target_signal.unsqueeze(1).repeat(1, l, 1).squeeze(-1)
            )
        else:
            target_signal = x_mod[:, :, self.target_idx]

        t = (
            torch.arange(l, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .repeat(b, 1)
        )

        if self.shortcut_type == "sine_hum":
            if not is_ood:
                phase_shift = target_signal * 0.5
                cue = self.amplitude * torch.sin(
                    2.0 * np.pi * self.freq * t + phase_shift
                )
                shortcut = (self.correlation_strength * target_signal) + (
                    (1.0 - self.correlation_strength) * cue
                )
            else:
                shifted_freq = self.freq * 3.0
                cue = self.amplitude * torch.sin(
                    2.0 * np.pi * shifted_freq * t + (np.pi / 2.0)
                )
                shortcut = (-self.correlation_strength * target_signal) + cue

            x_mod[:, :, self.channel_idx] += shortcut

        elif self.shortcut_type == "baseline_drift":
            if not is_ood:
                drift = (self.amplitude / l) * t * torch.sign(
                    target_signal + 1e-4
                )
            else:
                drift = -(self.amplitude / l) * t * torch.sign(
                    target_signal + 1e-4
                )
            x_mod[:, :, self.channel_idx] += drift

        return x_mod


def correlation_with_target(shortcut, target):
    """Pearson correlation between the injected shortcut and the target series.

    Implements the Phase-1 verification metric corr(S_t, Y): the poisoned
    training split must reach >= 90% and the OOD test split must drop
    below 10%. Accepts numpy arrays or torch tensors of equal length.
    """
    s = np.asarray(shortcut, dtype=np.float64).ravel()
    y = np.asarray(target, dtype=np.float64).ravel()
    if s.size != y.size:
        raise ValueError("shortcut and target must have the same length")
    if s.std() == 0.0 or y.std() == 0.0:
        return 0.0
    return float(np.corrcoef(s, y)[0, 1])
