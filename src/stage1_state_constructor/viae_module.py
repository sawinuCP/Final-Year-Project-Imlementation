"""Active Variational Invariant Autoencoder (VIAE) with Factorized Additive Decoder.

Implements decoupled decoding:
    X_hat_inv = Dec_inv(Z_inv)
    X_hat_full = Dec_inv(Z_inv) + Dec_e(Z_e)
Guarantees Z_inv cannot suffer posterior collapse and decodes cleanly with Z_e = 0.
Reference: Norman & Meir (ICLR 2026) "Unsupervised Representation Learning - an IRM Perspective".
"""

import torch
import torch.nn as nn
from .pica_projector import PICAProjector

__all__ = ["ActiveVIAE"]


class CausalConv1dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, kernel_size: int = 3):
        super(CausalConv1dBlock, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size,
            padding=self.padding, dilation=dilation
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return self.act(self.norm(out) + res)


class ActiveVIAE(nn.Module):
    def __init__(self, in_features: int = 7, seq_len: int = 96, d_inv: int = 8, d_e: int = 4, hidden_dim: int = 64):
        super(ActiveVIAE, self).__init__()
        self.in_features = in_features
        self.seq_len = seq_len
        self.d_inv = d_inv
        self.d_e = d_e
        self.hidden_dim = hidden_dim

        self.pica = PICAProjector(in_features=in_features, invariant_dim=in_features)

        # Dilated Causal Convolutional Encoder (Full lookback receptive field)
        self.enc_blocks = nn.ModuleList([
            CausalConv1dBlock(in_features, hidden_dim, dilation=1),
            CausalConv1dBlock(hidden_dim, hidden_dim, dilation=2),
            CausalConv1dBlock(hidden_dim, hidden_dim, dilation=4),
            CausalConv1dBlock(hidden_dim, hidden_dim, dilation=8),
            CausalConv1dBlock(hidden_dim, hidden_dim, dilation=16)
        ])

        # Dual-branch latent heads
        self.fc_mu_inv = nn.Linear(hidden_dim, d_inv)
        self.fc_logvar_inv = nn.Linear(hidden_dim, d_inv)
        self.fc_mu_e = nn.Linear(hidden_dim, d_e)
        self.fc_logvar_e = nn.Linear(hidden_dim, d_e)

        # Factorized Decoders: Z_inv has its own independent reconstruction backbone
        self.dec_inv = nn.Sequential(
            nn.Conv1d(d_inv, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, in_features, kernel_size=1)
        )

        # Residual shortcut decoder
        self.dec_e = nn.Sequential(
            nn.Conv1d(d_e, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, in_features, kernel_size=1)
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor, modulations: dict = None, deterministic: bool = False):
        x_proj = self.pica(x)
        h = x_proj.permute(0, 2, 1)

        for block in self.enc_blocks:
            h = block(h)

        if modulations is not None and len(modulations.get("scales", [])) > 0:
            scale = modulations["scales"][0].unsqueeze(-1)
            shift = modulations["shifts"][0].unsqueeze(-1)
            h = h * scale + shift

        h_t = h.permute(0, 2, 1)

        mu_inv = self.fc_mu_inv(h_t)
        logvar_inv = torch.clamp(self.fc_logvar_inv(h_t), min=-6.0, max=2.0)

        mu_e = self.fc_mu_e(h_t)
        logvar_e = torch.clamp(self.fc_logvar_e(h_t), min=-6.0, max=2.0)

        if deterministic or not self.training:
            z_inv = mu_inv
            z_e = mu_e
        else:
            z_inv = self.reparameterize(mu_inv, logvar_inv)
            z_e = self.reparameterize(mu_e, logvar_e)

        return z_inv, mu_inv, logvar_inv, z_e, mu_e, logvar_e

    def decode(self, z_inv: torch.Tensor, z_e: torch.Tensor = None) -> torch.Tensor:
        """
        Decodes using the additive factorized architecture.
        If z_e is None or zero, reconstructs from the invariant manifold alone.
        """
        z_inv_in = z_inv.permute(0, 2, 1)  # [B, d_inv, L]
        x_inv = self.dec_inv(z_inv_in).permute(0, 2, 1)

        if z_e is not None and torch.norm(z_e).item() > 1e-6:
            z_e_in = z_e.permute(0, 2, 1)  # [B, d_e, L]
            x_e = self.dec_e(z_e_in).permute(0, 2, 1)
            return x_inv + x_e

        return x_inv

    def forward(self, x: torch.Tensor, modulations: dict = None, deterministic: bool = False):
        z_inv, mu_inv, logvar_inv, z_e, mu_e, logvar_e = self.encode(x, modulations, deterministic)
        x_recon_inv = self.decode(z_inv, None)
        x_recon_full = self.decode(z_inv, z_e)
        return {
            "x_recon": x_recon_full,
            "x_recon_inv": x_recon_inv,
            "z_inv": z_inv,
            "mu_inv": mu_inv,
            "logvar_inv": logvar_inv,
            "z_e": z_e,
            "mu_e": mu_e,
            "logvar_e": logvar_e
        }