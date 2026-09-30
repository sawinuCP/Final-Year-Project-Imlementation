"""Phase 4 verification: Active Causal Guard & null-space weight repair."""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from src.stage1_state_constructor.hypernetwork import HypernetworkController
from src.stage1_state_constructor.viae_module import ActiveVIAE
from src.stage2_forecaster.hutchinson_radar import HutchinsonTopologicalRadar
from src.stage2_forecaster.mc_dropout_engine import AdaptiveMCDropoutEngine
from src.stage2_forecaster.patchtst_module import LatentPatchTST
from src.stage3_causal_guard.causal_guardrail import ActiveCausalGuard
from src.stage3_causal_guard.nullspace_patcher import NullSpacePatcher


def test_nullspace_projection_idempotence_and_symmetry():
    patcher = NullSpacePatcher()
    z_e = torch.randn(16, 4, dtype=torch.float32)
    p_inv = patcher.compute_nullspace_projection(z_e, target_dim=128, d_e=4)

    p_sq = torch.matmul(p_inv, p_inv)
    assert torch.norm(p_sq - p_inv, p="fro").item() < 1e-4
    assert torch.norm(p_inv.t() - p_inv, p="fro").item() < 1e-4


def test_nullspace_weight_patch_annihilation():
    patcher = NullSpacePatcher()
    z_e = torch.randn(16, 4, dtype=torch.float32)
    p_inv = patcher.compute_nullspace_projection(z_e, target_dim=128, d_e=4)

    linear_layer = nn.Linear(128, 96, bias=False)
    w_orig = linear_layer.weight.clone()
    w_repaired = patcher.patch_linear_weights(linear_layer, p_inv)

    _, residual_norm = patcher.verify_annihilation(w_repaired, z_e)
    assert residual_norm < 1e-4
    assert torch.norm(linear_layer.weight - w_orig, p="fro").item() == 0.0


def test_nullspace_repair_latency_smoke():
    patcher = NullSpacePatcher()
    z_e = torch.randn(16, 4, dtype=torch.float32)
    linear_layer = nn.Linear(128, 96, bias=False)

    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        p_inv = patcher.compute_nullspace_projection(z_e, target_dim=128, d_e=4)
        patcher.patch_linear_weights(linear_layer, p_inv)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    # CPU smoke budget (GPU target remains < 2 ms in spec)
    assert sum(latencies) / len(latencies) < 150.0


def test_nullspace_patcher_patchtst_head_dim():
    patcher = NullSpacePatcher()
    z_e = torch.randn(16, 96, 4, dtype=torch.float32)
    p_inv = patcher.compute_nullspace_projection(z_e, target_dim=704, d_e=4)
    assert p_inv.shape == (704, 704)
    linear = nn.Linear(704, 96, bias=False)
    w_rep = patcher.patch_linear_weights(linear, p_inv)
    _, residual = patcher.verify_annihilation(w_rep, z_e)
    assert residual < 1e-3


def test_stage3_guard_pipeline():
    batch_size = 16
    seq_len = 96
    pred_len = 96
    in_features = 7
    d_inv = 8
    d_e = 4
    hidden_dim = 64
    d_model = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hypernet = HypernetworkController(modulation_dims=[hidden_dim], hidden_dim=64).to(device)
    viae = ActiveVIAE(
        in_features=in_features, seq_len=seq_len, d_inv=d_inv, d_e=d_e, hidden_dim=hidden_dim
    ).to(device)
    forecaster = LatentPatchTST(
        seq_len=seq_len, pred_len=pred_len, d_inv=d_inv, patch_len=16, stride=8, d_model=d_model
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

    radar.tau_base = 1e-6
    guard.forward_stream(x_clean)
    repair_latencies = []
    res_repaired = None
    for _ in range(3):
        res_repaired = guard.forward_stream(x_clean)
        repair_latencies.append(res_repaired["repair_latency_ms"])

    assert res_repaired is not None
    assert res_repaired["repair_applied"]
    assert res_repaired["forecast"].shape == (batch_size, pred_len, in_features)
    if device.type == "cuda":
        assert min(repair_latencies) < 2.0
    else:
        assert min(repair_latencies) < 150.0
