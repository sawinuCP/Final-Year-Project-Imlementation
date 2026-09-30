"""Latent Time Series Forecasting (LatentTSF) Baseline.

Implements the two-stage latent state forecasting paradigm:
    1. Pre-train a continuous observation-to-state Autoencoder.
    2. Explicitly FREEZE all Autoencoder parameters (requires_grad = False).
    3. Train a downstream PatchTST forecaster purely within this frozen latent space.
Reference: Yang et al. (ICML 2026) "From Observations to States: Latent Time Series Forecasting".
"""

import torch
import torch.nn as nn
from src.stage2_forecaster.patchtst_module import LatentPatchTST

__all__ = ["LatentTSFAutoencoder", "LatentTSFModel"]


class LatentTSFAutoencoder(nn.Module):
    """
    Standard Point-wise Autoencoder for LatentTSF.
    Compacts raw observation space [B, L, D] to latent manifold [B, L, d_latent].
    """
    def __init__(self, in_features: int = 7, d_latent: int = 8, hidden_dim: int = 64):
        super(LatentTSFAutoencoder, self).__init__()
        self.in_features = in_features
        self.d_latent = d_latent

        self.encoder = nn.Sequential(
            nn.Conv1d(in_features, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, d_latent, kernel_size=1)
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(d_latent, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, in_features, kernel_size=1)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] -> [B, D, L] -> conv -> [B, d_latent, L] -> [B, L, d_latent]
        h = self.encoder(x.permute(0, 2, 1))
        return h.permute(0, 2, 1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, H, d_latent] -> [B, d_latent, H] -> conv -> [B, D, H] -> [B, H, D]
        x_recon = self.decoder(z.permute(0, 2, 1))
        return x_recon.permute(0, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


class LatentTSFModel(nn.Module):
    """
    LatentTSF end-to-end wrapper combining frozen Autoencoder and latent forecaster.
    """
    def __init__(
        self,
        in_features: int = 7,
        seq_len: int = 96,
        pred_len: int = 96,
        d_latent: int = 8,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64
    ):
        super(LatentTSFModel, self).__init__()
        self.in_features = in_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_latent = d_latent

        self.autoencoder = LatentTSFAutoencoder(in_features=in_features, d_latent=d_latent)
        self.latent_forecaster = LatentPatchTST(
            seq_len=seq_len,
            pred_len=pred_len,
            d_inv=d_latent,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model
        )

    def freeze_autoencoder(self):
        """Strictly freezes all autoencoder weights (LatentTSF invariant constraint)."""
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference pass:
            1. Encode observation into frozen state z = AE_enc(x)
            2. Forecast future state z_hat = Forecaster(z)
            3. Decode state to observation y_hat = AE_dec(z_hat)
        """
        with torch.no_grad():
            z = self.autoencoder.encode(x)

        z_hat = self.latent_forecaster(z)

        with torch.no_grad():
            y_hat = self.autoencoder.decode(z_hat)

        return y_hat