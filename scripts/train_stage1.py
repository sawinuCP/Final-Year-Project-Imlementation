"""Stage 1 training: Active VIAE + Hyper-CRIB on ETTh1."""

from _project_root import setup

setup()

import os

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from src.data_engine.dataset_loader import ETTh1Dataset
from src.data_engine.shortcut_injector import SyntheticShortcutInjector
from src.stage1_state_constructor.crib_loss import CRIBLoss
from src.stage1_state_constructor.hypernetwork import HypernetworkController
from src.stage1_state_constructor.viae_module import ActiveVIAE


def train_stage1():
    print("=" * 75)
    print("STAGE 1 TRAINING: Active VIAE + Hyper-CRIB on ETTh1")
    print("=" * 75)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Precision: FP32")

    train_dataset = ETTh1Dataset(
        root_path="data/raw/ETTh1.csv", flag="train", size=(96, 96), features="M"
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    in_features = 7
    seq_len = 96
    d_inv = 8
    d_e = 4
    hidden_dim = 64

    print("\nFitting PICA on real 6-month temporal chunks of ETTh1...")
    chunk_size = len(train_dataset.data_x) // 2
    x_chunk1 = torch.tensor(train_dataset.data_x[:chunk_size], dtype=torch.float32)
    x_chunk2 = torch.tensor(train_dataset.data_x[chunk_size:], dtype=torch.float32)

    viae = ActiveVIAE(
        in_features=in_features, seq_len=seq_len, d_inv=d_inv, d_e=d_e, hidden_dim=hidden_dim
    ).to(device)
    viae.pica.fit(x_chunk1, x_chunk2)
    print("PICA null-space matrix initialized.")

    hypernet = HypernetworkController(modulation_dims=[hidden_dim], hidden_dim=64).to(device)
    crib_loss_fn = CRIBLoss(d_inv=d_inv, gamma_consistency=1.0, lambda_res=0.5).to(device)
    injector = SyntheticShortcutInjector(shortcut_type="sine_hum", channel_idx=0, amplitude=2.0)

    optimizer = optim.Adam(
        list(viae.parameters()) + list(hypernet.parameters()), lr=1e-3, weight_decay=1e-5
    )

    epochs = 10
    print(f"\nTraining Active VIAE with Hyper-CRIB for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        viae.train()
        hypernet.train()
        total_loss_accum = 0.0

        for bx, by, domain_ids in train_loader:
            bx = bx.to(device)
            by = by.to(device)
            domain_ids = domain_ids.to(device)

            bx_poisoned = injector.inject_shortcut(bx, y=by, is_ood=False)

            log_beta = torch.empty(1, 1, device=device).uniform_(-2.0, 2.0)
            beta_val = torch.exp(log_beta).item()
            modulations = hypernet(torch.tensor([[beta_val]], device=device, dtype=torch.float32))

            out_clean = viae(bx_poisoned, modulations)
            noise = torch.randn_like(bx_poisoned) * 0.05
            out_perturbed = viae(bx_poisoned + noise, modulations)

            losses = crib_loss_fn(
                out_clean, out_perturbed, bx_poisoned, beta=beta_val, domain_labels=domain_ids
            )
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()

            total_loss_accum += losses["loss"].item()

        avg_loss = total_loss_accum / len(train_loader)
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] - Multi-Rate ELBO Loss: {avg_loss:.4f} | "
            f"Recon: {losses['recon_loss']:.4f} | Markov KL: {losses['markov_kl']:.4f}"
        )

    checkpoint_path = "checkpoints/stage1_viae.pt"
    torch.save(
        {
            "viae_state_dict": viae.state_dict(),
            "hypernet_state_dict": hypernet.state_dict(),
            "u_null": viae.pica.u_null,
        },
        checkpoint_path,
    )
    print(f"\nStage 1 checkpoint saved to {checkpoint_path}")

    print("\nGenerating t-SNE trajectory visualization...")
    viae.eval()
    with torch.no_grad():
        test_sample = (
            torch.tensor(train_dataset.data_x[:seq_len], dtype=torch.float32).unsqueeze(0).to(device)
        )
        mod_eval = hypernet(torch.tensor([[1.0]], device=device))
        z_inv, _, _, _, _, _ = viae.encode(test_sample, mod_eval)

        z_inv_np = z_inv.squeeze(0).cpu().numpy()
        raw_np = test_sample.squeeze(0).cpu().numpy()

        tsne = TSNE(n_components=2, perplexity=10, random_state=42)
        z_emb = tsne.fit_transform(z_inv_np)
        raw_emb = tsne.fit_transform(raw_np)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.scatter(raw_emb[:, 0], raw_emb[:, 1], c=range(seq_len), cmap="viridis")
        ax1.plot(raw_emb[:, 0], raw_emb[:, 1], "k--", alpha=0.3)
        ax1.set_title("Raw Observation Space (Latent Chaos)")

        ax2.scatter(z_emb[:, 0], z_emb[:, 1], c=range(seq_len), cmap="plasma")
        ax2.plot(z_emb[:, 0], z_emb[:, 1], "k--", alpha=0.3)
        ax2.set_title("Purified Invariant Latent Space Z_inv (Continuous Trajectory)")

        plt.tight_layout()
        plt.savefig("reports/figures/latent_chaos_resolved.png")
        plt.close()
        print("t-SNE figure saved to reports/figures/latent_chaos_resolved.png")


if __name__ == "__main__":
    train_stage1()
