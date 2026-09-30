"""Train literature competitor baselines on ETTh1 (ERM, LatentTSF, IRM-PatchTST)."""

from _project_root import setup

setup()

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.baselines.erm_patchtst import ERMPatchTST
from src.baselines.irm_patchtst import IRMPatchTST
from src.baselines.latent_tsf import LatentTSFModel
from src.data_engine.dataset_loader import ETTh1Dataset
from src.data_engine.shortcut_injector import SyntheticShortcutInjector


def train_baselines():
    print("=" * 80)
    print("TRAINING LITERATURE COMPETITOR BASELINES ON ETTh1 (FP32 Precision)")
    print("=" * 80)

    os.makedirs("checkpoints/baselines", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Hardware: {device} | Precision: torch.float32\n")

    train_dataset = ETTh1Dataset(
        root_path="data/raw/ETTh1.csv", flag="train", size=(96, 96), features="M"
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    injector = SyntheticShortcutInjector(shortcut_type="sine_hum", channel_idx=0, amplitude=2.0)

    in_features = 7
    seq_len = 96
    pred_len = 96

    print("--- [1/3] Training Standard ERM PatchTST ---")
    erm_model = ERMPatchTST(in_features=in_features, seq_len=seq_len, pred_len=pred_len).to(device)
    optimizer_erm = optim.Adam(erm_model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    erm_epochs = 5
    for epoch in range(1, erm_epochs + 1):
        erm_model.train()
        total_loss = 0.0
        for bx, by, _ in train_loader:
            bx, by = bx.to(device), by.to(device)
            bx_poisoned = injector.inject_shortcut(bx, y=by, is_ood=False)

            pred = erm_model(bx_poisoned)
            loss = criterion(pred, by)

            optimizer_erm.zero_grad()
            loss.backward()
            optimizer_erm.step()
            total_loss += loss.item()
        print(f"  Epoch [{epoch:02d}/{erm_epochs:02d}] - ERM MSE Loss: {total_loss / len(train_loader):.4f}")

    torch.save(erm_model.state_dict(), "checkpoints/baselines/erm_patchtst.pt")
    print("  Saved ERM checkpoint to checkpoints/baselines/erm_patchtst.pt\n")

    print("--- [2/3] Training LatentTSF (Yang et al., ICML 2026) ---")
    latent_tsf = LatentTSFModel(
        in_features=in_features, seq_len=seq_len, pred_len=pred_len, d_latent=8
    ).to(device)
    optimizer_ae = optim.Adam(latent_tsf.autoencoder.parameters(), lr=1e-3)

    ae_epochs = 4
    print("  Step A: Pre-training LatentTSF Autoencoder on raw sequences...")
    for epoch in range(1, ae_epochs + 1):
        latent_tsf.autoencoder.train()
        ae_loss = 0.0
        for bx, _, _ in train_loader:
            bx = bx.to(device)
            recon = latent_tsf.autoencoder(bx)
            loss = criterion(recon, bx)

            optimizer_ae.zero_grad()
            loss.backward()
            optimizer_ae.step()
            ae_loss += loss.item()
        print(f"    AE Epoch [{epoch:02d}/{ae_epochs:02d}] - Reconstruction MSE: {ae_loss / len(train_loader):.4f}")

    print("  Step B: Freezing Autoencoder weights (requires_grad = False)...")
    latent_tsf.freeze_autoencoder()

    optimizer_latent = optim.Adam(latent_tsf.latent_forecaster.parameters(), lr=1e-3)
    forecaster_epochs = 5
    print("  Step C: Training Latent PatchTST on frozen state space...")
    for epoch in range(1, forecaster_epochs + 1):
        latent_tsf.latent_forecaster.train()
        l_loss = 0.0
        for bx, by, _ in train_loader:
            bx, by = bx.to(device), by.to(device)
            bx_poisoned = injector.inject_shortcut(bx, y=by, is_ood=False)

            with torch.no_grad():
                z_x = latent_tsf.autoencoder.encode(bx_poisoned)
                z_y = latent_tsf.autoencoder.encode(by)

            z_pred = latent_tsf.latent_forecaster(z_x)
            loss = criterion(z_pred, z_y)

            optimizer_latent.zero_grad()
            loss.backward()
            optimizer_latent.step()
            l_loss += loss.item()
        print(
            f"    Forecaster Epoch [{epoch:02d}/{forecaster_epochs:02d}] - "
            f"Latent MSE: {l_loss / len(train_loader):.4f}"
        )

    torch.save(latent_tsf.state_dict(), "checkpoints/baselines/latent_tsf.pt")
    print("  Saved LatentTSF checkpoint to checkpoints/baselines/latent_tsf.pt\n")

    print("--- [3/3] Training IRM-PatchTST (FOIL / InvarNet Invariance Penalty) ---")
    irm_model = IRMPatchTST(in_features=in_features, seq_len=seq_len, pred_len=pred_len).to(device)
    optimizer_irm = optim.Adam(irm_model.parameters(), lr=1e-3, weight_decay=1e-5)

    irm_epochs = 5
    for epoch in range(1, irm_epochs + 1):
        irm_model.train()
        total_irm_loss = 0.0
        for bx, by, domain_ids in train_loader:
            bx, by, domain_ids = bx.to(device), by.to(device), domain_ids.to(device)
            bx_poisoned = injector.inject_shortcut(bx, y=by, is_ood=False)

            irm_res = irm_model.compute_irm_loss(bx_poisoned, by, domain_ids, lambda_irm=1.0)
            loss = irm_res["loss"]

            optimizer_irm.zero_grad()
            loss.backward()
            optimizer_irm.step()
            total_irm_loss += loss.item()
        print(
            f"  Epoch [{epoch:02d}/{irm_epochs:02d}] - Total IRM Loss: "
            f"{total_irm_loss / len(train_loader):.4f} (Risk: {irm_res['risk']:.4f}, "
            f"Penalty: {irm_res['penalty']:.6f})"
        )

    torch.save(irm_model.state_dict(), "checkpoints/baselines/irm_patchtst.pt")
    print("  Saved IRM-PatchTST checkpoint to checkpoints/baselines/irm_patchtst.pt\n")

    print("=" * 80)
    print("ALL COMPETITOR BASELINES TRAINED AND CHECKPOINTED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    train_baselines()
