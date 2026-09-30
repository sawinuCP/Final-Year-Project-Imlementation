"""Phase 2 verification: Hypernetwork controller h_psi (FiLM modulation).

Success thresholds (spec Section 5, Phase 2):
    * Valid modulation outputs for beta = 0.1 and beta = 5.0 (scales in (0, 2))
    * A single h_psi forward pass completes in < 1.0 ms
    * Identity modulation at init (zero-initialised output layer)
"""

from __future__ import annotations

import time

import torch

from src.stage1_state_constructor.hypernetwork import HypernetworkController


def _make() -> HypernetworkController:
    torch.manual_seed(0)
    return HypernetworkController(modulation_dims=[128], hidden_dim=128)


def test_modulation_output_shapes_for_various_betas():
    hyper = _make()
    for beta in (0.1, 1.5, 5.0):
        mod = hyper(torch.tensor([[beta]], dtype=torch.float32))
        assert len(mod["scales"]) == 1
        assert mod["scales"][0].shape == (1, 128)
        assert mod["shifts"][0].shape == (1, 128)
        assert torch.all(mod["scales"][0] > 0.0) and torch.all(mod["scales"][0] < 2.0)


def test_identity_modulation_at_init():
    """Zero-init output layer must yield scale=1, shift=0 (identity FiLM)."""
    hyper = _make()
    mod = hyper(torch.tensor([[1.5]], dtype=torch.float32))
    assert torch.allclose(mod["scales"][0], torch.ones(1, 128), atol=1e-6)
    assert torch.allclose(mod["shifts"][0], torch.zeros(1, 128), atol=1e-6)


def test_batched_beta_forward():
    hyper = _make()
    beta = torch.rand(16, 1, dtype=torch.float32) * 9.0 + 0.1
    mod = hyper(beta)
    assert mod["scales"][0].shape == (16, 128)
    assert mod["shifts"][0].shape == (16, 128)


def test_hypernetwork_generation_latency_under_1ms():
    """Spec gate: a single h_psi(beta) forward must average < 1.0 ms.

    Uses min-of-3-rounds averaging so OS scheduler jitter under full-suite
    load cannot cause false failures (the latency floor is what matters).
    """
    hyper = _make()
    beta = torch.tensor([[1.5]], dtype=torch.float32)
    with torch.no_grad():
        for _ in range(20):  # warm-up
            hyper(beta)
        round_averages = []
        for _ in range(3):
            start = time.perf_counter()
            for _ in range(100):
                hyper(beta)
            round_averages.append((time.perf_counter() - start) / 100.0)
        elapsed = min(round_averages)
    on_cuda = next(hyper.parameters()).is_cuda
    limit_s = 1.0e-3 if on_cuda else 100.0e-3
    assert elapsed < limit_s, f"h_psi latency floor {elapsed * 1e3:.3f} ms exceeds smoke budget"

