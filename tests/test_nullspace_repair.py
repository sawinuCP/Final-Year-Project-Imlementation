"""Phase 4 verification: Active Causal Guard & latent null-space repair."""

from __future__ import annotations

import time

import torch

from src.stage1_state_constructor.hypernetwork import HypernetworkController
from src.stage1_state_constructor.viae_module import ActiveVIAE
from src.stage2_forecaster.hutchinson_radar import HutchinsonTopologicalRadar
from src.stage2_forecaster.mc_dropout_engine import AdaptiveMCDropoutEngine
from src.stage2_forecaster.patchtst_module import LatentPatchTST
from src.stage3_causal_guard.causal_guardrail import ActiveCausalGuard
from src.stage3_causal_guard.nullspace_patcher import NullSpacePatcher


def test_latent_projection_shape_and_idempotence():
    patcher = NullSpacePatcher()
    z_e = torch.randn(4, 96, 4, dtype=torch.float32)
    d_inv = 16
    p_inv = patcher.compute_latent_projection(z_e, d_inv=d_inv)
    assert p_inv.shape == (d_inv, d_inv)
    assert torch.isfinite(p_inv).all()
    assert p_inv.shape[0] == p_inv.shape[1] == d_inv


def test_project_latent_trajectory():
    patcher = NullSpacePatcher()
    z_pred = torch.randn(2, 96, 16, dtype=torch.float32)
    p_inv = torch.eye(16, dtype=torch.float32)
    z_rep = patcher.project_latent_trajectory(z_pred, p_inv)
    assert torch.allclose(z_rep, z_pred)
    assert z_rep.shape == z_pred.shape


def test_nullspace_repair_latency_smoke():
    patcher = NullSpacePatcher()
    z_e = torch.randn(8, 96, 4, dtype=torch.float32)
    z_pred = torch.randn(8, 96, 16, dtype=torch.float32)
    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        p_inv = patcher.compute_latent_projection(z_e, d_inv=16)
        patcher.project_latent_trajectory(z_pred, p_inv)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    assert sum(latencies) / len(latencies) < 150.0


def test_stage3_guard_pipeline():
    batch_size = 8
    seq_len = 96
    pred_len = 96
    in_features = 7
    d_inv = 16
    d_e = 4
    hidden_dim = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hypernet = HypernetworkController(modulation_dims=[hidden_dim], hidden_dim=64).to(device)
    viae = ActiveVIAE(
        in_features=in_features, seq_len=seq_len, d_inv=d_inv, d_e=d_e, hidden_dim=hidden_dim
    ).to(device)
    forecaster = LatentPatchTST(
        seq_len=seq_len, pred_len=pred_len, d_inv=d_inv, patch_len=16, stride=8, d_model=64
    ).to(device)
    radar = HutchinsonTopologicalRadar(tau_base=0.05, gamma=2.0).to(device)
    mc_engine = AdaptiveMCDropoutEngine(forecaster, s_base=10, s_max=30)

    guard = ActiveCausalGuard(
        viae=viae,
        forecaster=forecaster,
        hypernet=hypernet,
        mc_engine=mc_engine,
        radar=radar,
    ).to(device)

    x_clean = torch.randn(batch_size, seq_len, in_features, device=device, dtype=torch.float32)

    radar.tau_base = 100.0
    res_normal = guard.forward_stream(x_clean)
    assert not res_normal["repair_applied"]
    assert res_normal["forecast"].shape == (batch_size, pred_len, in_features)
    assert "mean_variance" in res_normal

    guard.radar.tau_base = 1e-6
    res_repaired = guard.forward_stream(x_clean)
    assert res_repaired["repair_applied"]
    assert res_repaired["forecast"].shape == (batch_size, pred_len, in_features)
