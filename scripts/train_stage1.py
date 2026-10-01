"""Stage 1 Training: Active VIAE + Hyper-CRIB on ETTh1."""

from _project_root import setup

setup()

import os

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from src.data_engine.dataset_loader import ETTh1Dataset
from src.data_engine.shortcut_injector import build_shortcut_injector
from src.stage1_state_constructor.hypernetwork import HypernetworkController
from src.stage1_state_constructor.viae_module import ActiveVIAE
from src.stage1_state_constructor.crib_loss import CRIBLoss


def train_stage1():
    print("=" * 80)
    print("STAGE 1: Training Active VIAE with Factorized Invariant Decoder")
    print("=" * 80)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Precision: torch.float32\n")

    train_dataset = ETTh1Dataset(root_path="data/raw/ETTh1.csv", flag='train', size=(96, 96), features="M")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    in_features, seq_len, d_inv, d_e, hidden_dim = 7, 96, 8, 4, 64

    # Fit PICA
    print("Fitting PICA on ETTh1 temporal chunks...")
    chunk_size = len(train_dataset.data_x) // 2
    x_chunk1 = torch.tensor(train_dataset.data_x[:chunk_size], dtype=torch.float32)
    x_chunk2 = torch.tensor(train_dataset.data_x[chunk_size:], dtype=torch.float32)

    viae = ActiveVIAE(in_features=in_features, seq_len=seq_len, d_inv=d_inv, d_e=d_e, hidden_dim=hidden_dim).to(device)
    viae.pica.fit(x_chunk1, x_chunk2)

    hypernet = HypernetworkController(modulation_dims=[hidden_dim], hidden_dim=64).to(device)
    crib_loss_fn = CRIBLoss(d_inv=d_inv, gamma_consistency=0.5, lambda_res=0.2).to(device)
    injector = build_shortcut_injector()

    optimizer = optim.Adam(list(viae.parameters()) + list(hypernet.parameters()), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    epochs = 20
    print(f"Training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        viae.train()
        hypernet.train()
        total_loss, total_recon_inv = 0.0, 0.0

        for bx, by, domain_ids in train_loader:
            bx, by, domain_ids = bx.to(device), by.to(device), domain_ids.to(device)
            bx_poisoned = injector.inject_shortcut(bx, y=by, is_ood=False)

            log_beta = torch.empty(1, 1, device=device).uniform_(-2.0, 2.0)
            beta_val = torch.exp(log_beta).item()
            modulations = hypernet(torch.tensor([[beta_val]], device=device, dtype=torch.float32))

            out_clean = viae(bx_poisoned, modulations)
            noise = torch.randn_like(bx_poisoned) * 0.05
            out_perturbed = viae(bx_poisoned + noise, modulations)

            losses = crib_loss_fn(out_clean, out_perturbed, bx_poisoned, beta=beta_val, domain_labels=domain_ids)
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()

            total_loss += losses["loss"].item()
            total_recon_inv += losses["recon_inv"].item()

        avg_loss = total_loss / len(train_loader)
        avg_recon_inv = total_recon_inv / len(train_loader)
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} | Recon_Inv MSE: {avg_recon_inv:.4f} | LR: {lr:.6f}")
        scheduler.step(avg_loss)

    # Save checkpoint
    torch.save({
        "viae_state_dict": viae.state_dict(),
        "hypernet_state_dict": hypernet.state_dict(),
        "u_null": viae.pica.u_null
    }, "checkpoints/stage1_viae.pt")
    print("\nStage 1 checkpoint saved to checkpoints/stage1_viae.pt")


if __name__ == "__main__":
    train_stage1()