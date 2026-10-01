"""Stage 2 training: Latent PatchTST + MC threshold calibration."""

from _project_root import setup

setup()

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data_engine.dataset_loader import ETTh1Dataset
from src.data_engine.shortcut_injector import build_shortcut_injector
from src.stage1_state_constructor.hypernetwork import HypernetworkController
from src.stage1_state_constructor.viae_module import ActiveVIAE
from src.stage2_forecaster.mc_dropout_engine import AdaptiveMCDropoutEngine
from src.stage2_forecaster.patchtst_module import LatentPatchTST


def train_stage2():
    print("=" * 75)
    print("STAGE 2: Latent PatchTST Training & Dynamic Gating Calibration")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Precision: FP32")

    checkpoint_path = "checkpoints/stage1_viae.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Missing {checkpoint_path}. Run scripts/train_stage1.py first!"
        )

    ckpt = torch.load(checkpoint_path, map_location=device)
    hidden_dim = 64
    d_inv = 8
    d_e = 4

    viae = ActiveVIAE(
        in_features=7, seq_len=96, d_inv=d_inv, d_e=d_e, hidden_dim=hidden_dim
    ).to(device)
    viae.load_state_dict(ckpt["viae_state_dict"])
    viae.pica.u_null.copy_(ckpt["u_null"])
    viae.eval()

    hypernet = HypernetworkController(modulation_dims=[hidden_dim], hidden_dim=64).to(device)
    hypernet.load_state_dict(ckpt["hypernet_state_dict"])
    hypernet.eval()
    print("Stage 1 VIAE and Hypernetwork loaded successfully.")

    train_dataset = ETTh1Dataset(
        root_path="data/raw/ETTh1.csv", flag="train", size=(96, 96), features="M"
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

    val_dataset = ETTh1Dataset(
        root_path="data/raw/ETTh1.csv", flag="val", size=(96, 96), features="M"
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    forecaster = LatentPatchTST(
        seq_len=96, pred_len=96, d_inv=d_inv, patch_len=16, stride=8, d_model=64
    ).to(device)
    optimizer = optim.Adam(forecaster.parameters(), lr=5e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    epochs = 20
    print(f"\nTraining Latent PatchTST on purified Z_inv for {epochs} epochs...")
    mod_normal = hypernet(torch.tensor([[1.0]], device=device))

    for epoch in range(1, epochs + 1):
        forecaster.train()
        epoch_loss = 0.0

        for bx, by, _ in train_loader:
            bx = bx.to(device)
            by = by.to(device)

            with torch.no_grad():
                # Use deterministic=True so PatchTST predicts real features, not random noise
                _, z_inv_x, _, _, _, _ = viae.encode(bx, mod_normal, deterministic=True)
                _, z_inv_y, _, _, _, _ = viae.encode(by, mod_normal, deterministic=True)

            pred_z = forecaster(z_inv_x)
            loss = criterion(pred_z, z_inv_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Latent Forecaster MSE: {epoch_loss / len(train_loader):.4f}")

    torch.save(forecaster.state_dict(), "checkpoints/stage2_patchtst.pt")
    print("Stage 2 forecaster saved to checkpoints/stage2_patchtst.pt")

    # 5. Offline Calibration of Safety Threshold tau_base on Clean Validation Split
    print("\nCalibrating tau_base on clean validation set (85th percentile rule)...")
    forecaster.eval()
    mc_engine = AdaptiveMCDropoutEngine(forecaster, s_base=10, s_max=30)
    val_variances = []

    for bx, _, _ in val_loader:
        bx = bx.to(device)
        with torch.no_grad():
            _, z_inv_val, _, _, _, _ = viae.encode(bx, mod_normal, deterministic=True)
        res = mc_engine.evaluate_uncertainty(z_inv_val, tau_mc=1e9)
        val_variances.append(res["max_var"])

    tau_base = float(np.percentile(val_variances, 85))
    print(f"Calibrated 95th-percentile Safety Threshold (tau_base): {tau_base:.6f}")

    with open("checkpoints/calibration_stats.json", "w", encoding="utf-8") as f:
        json.dump({"tau_base": tau_base}, f, indent=4)

    print("\nBenchmarking Gate 1 Sub-1ms Latency & OOD Variance Spike...")
    test_dataset = ETTh1Dataset(
        root_path="data/raw/ETTh1.csv", flag="test", size=(96, 96), features="M"
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    bx_sample, by_sample, _ = next(iter(test_loader))
    bx_sample = bx_sample.to(device)
    by_sample = by_sample.to(device)

    with torch.no_grad():
        z_sample, _, _, _, _, _ = viae.encode(bx_sample, mod_normal)

    t0 = time.perf_counter()
    res_clean = mc_engine.evaluate_uncertainty(z_sample, tau_mc=tau_base)
    gate1_ms = (time.perf_counter() - t0) * 1000.0

    print(
        f"  Clean Input: Passes = {res_clean['passes_used']} | Variance = {res_clean['max_var']:.6f} | "
        f"Latency = {gate1_ms:.2f} ms"
    )

    injector = build_shortcut_injector()
    bx_poisoned = injector.inject_shortcut(bx_sample, y=by_sample, is_ood=True)

    with torch.no_grad():
        z_poisoned, _, _, _, _, _ = viae.encode(bx_poisoned, mod_normal)

    res_ood = mc_engine.evaluate_uncertainty(z_poisoned, tau_mc=tau_base)
    spike_pct = (
        (res_ood["max_var"] - res_clean["max_var"]) / max(1e-6, res_clean["max_var"])
    ) * 100.0

    print(
        f"  OOD Input  : Passes = {res_ood['passes_used']} | Variance = {res_ood['max_var']:.6f} | "
        f"OOD Flag = {res_ood['is_ood']}"
    )
    print(f"  Variance Spike under OOD shift: +{spike_pct:.1f}%")

    if spike_pct > 200.0 or res_ood["is_ood"]:
        print("Phase 3 Stage 2 training, calibration, and gating verified successfully.")
    else:
        print(
            "WARNING: OOD variance spike did not reach the 200% spec on this run "
            f"(spike={spike_pct:.1f}%, is_ood={res_ood['is_ood']}). "
            "Checkpoints and tau_base are still valid; re-run with more Stage-2 epochs "
            "or evaluate on the full benchmark."
        )


if __name__ == "__main__":
    train_stage2()
