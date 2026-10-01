"""Stage 2 Training: Latent PatchTST & Calibrated Gating."""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from src.data_engine.dataset_loader import ETTh1Dataset
from src.data_engine.shortcut_injector import SyntheticShortcutInjector
from src.stage1_state_constructor.hypernetwork import HypernetworkController
from src.stage1_state_constructor.viae_module import ActiveVIAE
from src.stage2_forecaster.patchtst_module import LatentPatchTST
from src.stage2_forecaster.mc_dropout_engine import AdaptiveMCDropoutEngine


def train_stage2():
    print("=" * 80)
    print("STAGE 2: Training Latent PatchTST & Calibrating Dynamic Gating")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Precision: torch.float32\n")

    ckpt = torch.load("checkpoints/stage1_viae.pt", map_location=device)
    hidden_dim, d_inv, d_e = 64, 8, 4

    viae = ActiveVIAE(in_features=7, seq_len=96, d_inv=d_inv, d_e=d_e, hidden_dim=hidden_dim).to(device)
    viae.load_state_dict(ckpt["viae_state_dict"])
    viae.pica.u_null.copy_(ckpt["u_null"])
    viae.eval()

    hypernet = HypernetworkController(modulation_dims=[hidden_dim], hidden_dim=64).to(device)
    hypernet.load_state_dict(ckpt["hypernet_state_dict"])
    hypernet.eval()

    train_dataset = ETTh1Dataset(root_path="data/raw/ETTh1.csv", flag='train', size=(96, 96), features="M")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_dataset = ETTh1Dataset(root_path="data/raw/ETTh1.csv", flag='val', size=(96, 96), features="M")
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    injector = SyntheticShortcutInjector(shortcut_type="sine_hum", channel_idx=-1, amplitude=2.0)
    forecaster = LatentPatchTST(seq_len=96, pred_len=96, d_inv=d_inv, patch_len=16, stride=8, d_model=64, dropout=0.15).to(device)
    optimizer = optim.Adam(forecaster.parameters(), lr=5e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    epochs = 15
    print(f"Training Latent PatchTST on Z_inv for {epochs} epochs...")
    mod_normal = hypernet(torch.tensor([[1.0]], device=device))

    for epoch in range(1, epochs + 1):
        forecaster.train()
        epoch_loss = 0.0

        for bx, by, _ in train_loader:
            bx, by = bx.to(device), by.to(device)
            # Ingest shortcut-poisoned data: VIAE must extract clean Z_inv
            bx_p = injector.inject_shortcut(bx, y=by, is_ood=False)

            with torch.no_grad():
                z_x, _, _, _, _, _ = viae.encode(bx_p, mod_normal, deterministic=True)
                z_y, _, _, _, _, _ = viae.encode(by, mod_normal, deterministic=True)

            pred_z = forecaster(z_x)
            loss = criterion(pred_z, z_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Forecaster Latent MSE: {epoch_loss / len(train_loader):.4f}")

    torch.save(forecaster.state_dict(), "checkpoints/stage2_patchtst.pt")
    print("\nSaved Stage 2 checkpoint to checkpoints/stage2_patchtst.pt")

    # Calibration
    print("\nCalibrating tau_base on validation split (80th percentile)...")
    forecaster.eval()
    mc_engine = AdaptiveMCDropoutEngine(forecaster, s_base=10, s_max=30)
    val_variances = []

    for bx, _, _ in val_loader:
        bx = bx.to(device)
        with torch.no_grad():
            z_val, _, _, _, _, _ = viae.encode(bx, mod_normal, deterministic=True)
        res = mc_engine.evaluate_uncertainty(z_val, tau_mc=1e9)
        val_variances.append(res["max_var"])

    tau_base = float(np.percentile(val_variances, 80))
    print(f"Calibrated tau_base: {tau_base:.6f}")
    with open("checkpoints/calibration_stats.json", "w") as f:
        json.dump({"tau_base": tau_base}, f, indent=4)


if __name__ == "__main__":
    train_stage2()