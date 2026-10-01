"""Phase 1 verification: synthetic shortcut injection engine.

Mechanics covered here (runnable now):
    * shape/dtype preservation (FP32); non-target channels remain clean
    * sine hum: standard frequency ID; OOD = 3x frequency + pi/2 phase shift
    * baseline drift: OOD slope inversion with equal magnitude
    * Pearson correlation helper correctness

Spec Section-5 correlation gates (corr >= 90% train / <= 10% OOD against Y)
stay skipped until the target-aligned injection loop is wired into the
dataset pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data_engine.shortcut_injector import (
    SyntheticShortcutInjector,
    build_shortcut_injector,
    correlation_with_target,
)

B, L, D = 4, 96, 7


@pytest.fixture()
def clean_batch():
    g = torch.Generator().manual_seed(42)
    return torch.randn(B, L, D, generator=g, dtype=torch.float32)


def test_injection_preserves_shape_and_fp32_dtype(clean_batch):
    out = SyntheticShortcutInjector("sine_hum").inject_shortcut(clean_batch)
    assert out.shape == clean_batch.shape
    assert out.dtype == torch.float32


def test_injection_does_not_mutate_input(clean_batch):
    original = clean_batch.clone()
    SyntheticShortcutInjector("sine_hum").inject_shortcut(clean_batch)
    assert torch.equal(clean_batch, original)


def test_untouched_channels_remain_clean(clean_batch):
    injector = SyntheticShortcutInjector("sine_hum", channel_idx=2)
    out = injector.inject_shortcut(clean_batch)
    assert torch.equal(out[:, :, 0], clean_batch[:, :, 0])
    assert torch.equal(out[:, :, 5], clean_batch[:, :, 5])


def _expected_sine_shortcut(
    inj: SyntheticShortcutInjector,
    target_signal: torch.Tensor,
    t: torch.Tensor,
    is_ood: bool,
) -> torch.Tensor:
    if not is_ood:
        phase_shift = target_signal * 0.5
        cue = inj.amplitude * torch.sin(2.0 * np.pi * inj.freq * t + phase_shift)
        return (inj.correlation_strength * target_signal) + (
            (1.0 - inj.correlation_strength) * cue
        )
    shifted_freq = inj.freq * inj.ood_freq_multiplier
    cue = inj.amplitude * torch.sin(
        2.0 * np.pi * shifted_freq * t + (np.pi / 2.0)
    )
    return (-inj.correlation_strength * target_signal) + cue


def test_sine_hum_id_signal_matches_formula(clean_batch):
    inj = SyntheticShortcutInjector(
        "sine_hum", channel_idx=0, amplitude=1.5, freq=0.2, correlation_strength=0.92
    )
    y = clean_batch[:, -96:, :]
    out = inj.inject_shortcut(clean_batch, y=y, is_ood=False)
    b, seq_len, _ = clean_batch.shape
    t = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).repeat(b, 1)
    target_signal = torch.mean(y[:, :, inj.target_idx], dim=1, keepdim=True)
    target_signal = target_signal.unsqueeze(1).repeat(1, seq_len, 1).squeeze(-1)
    expected = clean_batch[:, :, 0] + _expected_sine_shortcut(
        inj, target_signal, t, is_ood=False
    )
    assert torch.allclose(out[:, :, 0], expected, atol=1e-6)


def test_sine_hum_ood_triples_frequency_and_shifts_phase(clean_batch):
    inj = SyntheticShortcutInjector(
        "sine_hum", channel_idx=0, amplitude=1.5, freq=0.2, correlation_strength=0.92
    )
    y = clean_batch[:, -96:, :]
    iid = inj.inject_shortcut(clean_batch, y=y, is_ood=False)
    ood = inj.inject_shortcut(clean_batch, y=y, is_ood=True)

    b, seq_len, _ = clean_batch.shape
    t = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).repeat(b, 1)
    target_signal = torch.mean(y[:, :, inj.target_idx], dim=1, keepdim=True)
    target_signal = target_signal.unsqueeze(1).repeat(1, seq_len, 1).squeeze(-1)
    expected = clean_batch[:, :, 0] + _expected_sine_shortcut(
        inj, target_signal, t, is_ood=True
    )
    assert torch.allclose(ood[:, :, 0], expected, atol=1e-6)
    assert not torch.allclose(ood[:, :, 0], iid[:, :, 0], atol=1e-3)


def test_baseline_drift_ood_inverts_slope(clean_batch):
    inj = SyntheticShortcutInjector("baseline_drift", channel_idx=1, amplitude=1.5)
    y = clean_batch[:, -96:, :]
    iid = inj.inject_shortcut(clean_batch, y=y, is_ood=False)
    ood = inj.inject_shortcut(clean_batch, y=y, is_ood=True)

    b, seq_len, _ = clean_batch.shape
    t = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).repeat(b, 1)
    target_signal = torch.mean(y[:, :, inj.target_idx], dim=1, keepdim=True)
    target_signal = target_signal.unsqueeze(1).repeat(1, seq_len, 1).squeeze(-1)
    sign = torch.sign(target_signal + 1e-4)
    drift_id = (1.5 / seq_len) * t * sign
    drift_ood = -(1.5 / seq_len) * t * sign
    assert torch.allclose(iid[:, :, 1], clean_batch[:, :, 1] + drift_id, atol=1e-6)
    assert torch.allclose(ood[:, :, 1], clean_batch[:, :, 1] + drift_ood, atol=1e-6)


def test_correlation_helper_known_signals():
    t = np.arange(256)
    s = np.sin(2 * np.pi * 0.05 * t)
    assert correlation_with_target(s, s) == pytest.approx(1.0)
    assert correlation_with_target(s, -s) == pytest.approx(-1.0)
    rng = np.random.default_rng(0)
    assert abs(correlation_with_target(s, rng.normal(size=256))) < 0.3


def test_correlation_helper_zero_variance_is_zero():
    assert correlation_with_target(np.zeros(10), np.arange(10)) == 0.0


def test_correlation_helper_length_mismatch_raises():
    with pytest.raises(ValueError):
        correlation_with_target(np.zeros(10), np.zeros(5))


def test_build_shortcut_injector_matches_yaml():
    inj = build_shortcut_injector()
    assert inj.shortcut_type == "sine_hum"
    assert inj.channel_idx == -1
    assert inj.amplitude == pytest.approx(2.0, rel=1e-3)


def test_from_yaml_sine_hum_config():
    inj = SyntheticShortcutInjector.from_yaml("configs/shortcuts/sine_hum.yaml")
    assert inj.shortcut_type == "sine_hum"
    assert inj.channel_idx == -1
    assert inj.amplitude == pytest.approx(2.0, rel=1e-3)
    assert inj.freq == pytest.approx(0.1, rel=1e-3)
    assert inj.correlation_strength == pytest.approx(0.92, rel=1e-3)
    assert inj.ood_freq_multiplier == pytest.approx(3.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Target-aligned correlation gates (reference-signal Pearson checks)
# ---------------------------------------------------------------------------

def _target_reference_series(y: torch.Tensor, seq_len: int, target_idx: int = -1) -> torch.Tensor:
    """Per-sample mean target over horizon, broadcast to length ``seq_len`` (matches injector)."""
    if y.shape[-1] == 1:
        ref = torch.mean(y, dim=1, keepdim=True)
    else:
        ref = torch.mean(y[:, :, target_idx], dim=1, keepdim=True)
    return ref.unsqueeze(1).repeat(1, seq_len, 1).squeeze(-1)


def test_sine_hum_train_correlation_ge_90pct(clean_batch):
    """ID shortcut should correlate with the target reference signal used for alignment."""
    inj = SyntheticShortcutInjector(
        "sine_hum", channel_idx=0, amplitude=1.5, freq=0.2, correlation_strength=0.92
    )
    b, seq_len, _ = clean_batch.shape
    y = torch.linspace(-2.0, 2.0, b, dtype=torch.float32).view(b, 1, 1).expand(b, seq_len, 7)
    out = inj.inject_shortcut(clean_batch, y=y, is_ood=False)
    shortcut = (out - clean_batch)[:, :, 0].reshape(-1).numpy()
    ref = _target_reference_series(y, seq_len).reshape(-1).numpy()
    assert correlation_with_target(shortcut, ref) >= 0.90


def test_sine_hum_ood_correlation_le_10pct(clean_batch):
    inj = SyntheticShortcutInjector(
        "sine_hum", channel_idx=0, amplitude=1.5, freq=0.2, correlation_strength=0.92
    )
    b, seq_len, _ = clean_batch.shape
    y = torch.linspace(-2.0, 2.0, b, dtype=torch.float32).view(b, 1, 1).expand(b, seq_len, 7)
    out = inj.inject_shortcut(clean_batch, y=y, is_ood=True)
    shortcut = (out - clean_batch)[:, :, 0].reshape(-1).numpy()
    ref = _target_reference_series(y, seq_len).reshape(-1).numpy()
    assert correlation_with_target(shortcut, ref) <= -0.50


def test_baseline_drift_ood_correlation_breaks(clean_batch):
    inj = SyntheticShortcutInjector("baseline_drift", channel_idx=1, amplitude=1.5)
    b, seq_len, _ = clean_batch.shape
    y = torch.sign(torch.linspace(-2.0, 2.0, b, dtype=torch.float32)).view(b, 1, 1).expand(b, seq_len, 7)
    iid = inj.inject_shortcut(clean_batch, y=y, is_ood=False)
    ood = inj.inject_shortcut(clean_batch, y=y, is_ood=True)
    s_iid = (iid - clean_batch)[:, :, 1].reshape(-1).numpy()
    s_ood = (ood - clean_batch)[:, :, 1].reshape(-1).numpy()
    assert not np.allclose(s_iid, s_ood)
    assert correlation_with_target(s_iid, s_ood) <= -0.99


@pytest.mark.skip(reason="Requires the ERM PatchTST baseline harness (Phase 1 remainder)")
def test_erm_patchtst_ood_degradation_gt_40pct():
    """ERM baseline must show > 40% MSE degradation on the OOD test set."""

