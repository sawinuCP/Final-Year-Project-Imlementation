"""Active Variational Invariant Autoencoder (VIAE)."""

import torch
import torch.nn as nn
from .pica_projector import PICAProjector

__all__ = ["ActiveVIAE"]


class CausalConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, dilation: int):
        super(CausalConvBlock, self).__init__()
        self.padding = (3 - 1) * dilation
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=3, padding=self.padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_c)
        self.act = nn.GELU()
        self.res = nn.Conv1d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.res(x)
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return self.act(self.norm(out) + res)


class ActiveVIAE(nn.Module):
    def __init__(self, in_features: int = 7, seq_len: int = 96, d_inv: int = 16, d_e: int = 4, hidden_dim: int = 64):
        super(ActiveVIAE, self).__init__()
        self.in_features = in_features
        self.seq_len = seq_len
        self.d_inv = d_inv
        self.d_e = d_e
        self.hidden_dim = hidden_dim

        # PICA Projector: 6 invariant dims, 1 spurious dim
        self.pica = PICAProjector(in_features=in_features, invariant_dim=6)

        # Invariant Encoder Stack (processes invariant features)
        self.enc_inv = nn.Sequential(
            CausalConvBlock(in_features, hidden_dim, dilation=1),
            CausalConvBlock(hidden_dim, hidden_dim, dilation=2),
            CausalConvBlock(hidden_dim, hidden_dim, dilation=4),
            CausalConvBlock(hidden_dim, hidden_dim, dilation=8)
        )
        self.fc_mu_inv = nn.Linear(hidden_dim, d_inv)
        self.fc_logvar_inv = nn.Linear(hidden_dim, d_inv)

        # Environmental Shortcut Encoder Stack
        self.enc_e = nn.Sequential(
            nn.Conv1d(in_features, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1)
        )
        self.fc_mu_e = nn.Linear(hidden_dim // 2, d_e)
        self.fc_logvar_e = nn.Linear(hidden_dim // 2, d_e)

        # High-Fidelity Decoders
        self.dec_inv = nn.Sequential(
            nn.Conv1d(d_inv, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, in_features, kernel_size=1)
        )

        self.dec_e = nn.Sequential(
            nn.Conv1d(d_e, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, in_features, kernel_size=1)
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def encode(self, x: torch.Tensor, modulations: dict = None, deterministic: bool = False):
        # 1. Project through PICA
        x_inv_proj = self.pica.project_invariant(x)
        x_spur_proj = self.pica.project_spurious(x)

        # 2. Encode Invariant Path
        h_inv = self.enc_inv(x_inv_proj.permute(0, 2, 1))
        if modulations is not None and len(modulations.get("scales", [])) > 0:
            scale = modulations["scales"][0].unsqueeze(-1)
            shift = modulations["shifts"][0].unsqueeze(-1)
            h_inv = h_inv * scale + shift
        h_inv_t = h_inv.permute(0, 2, 1)

        mu_inv = self.fc_mu_inv(h_inv_t)
        logvar_inv = torch.clamp(self.fc_logvar_inv(h_inv_t), min=-6.0, max=2.0)

        # 3. Encode Environmental Path
        h_e = self.enc_e(x_spur_proj.permute(0, 2, 1)).permute(0, 2, 1)
        mu_e = self.fc_mu_e(h_e)
        logvar_e = torch.clamp(self.fc_logvar_e(h_e), min=-6.0, max=2.0)

        if deterministic or not self.training:
            z_inv, z_e = mu_inv, mu_e
        else:
            z_inv = self.reparameterize(mu_inv, logvar_inv)
            z_e = self.reparameterize(mu_e, logvar_e)

        return z_inv, mu_inv, logvar_inv, z_e, mu_e, logvar_e

    def decode(self, z_inv: torch.Tensor, z_e: torch.Tensor = None) -> torch.Tensor:
        x_inv = self.dec_inv(z_inv.permute(0, 2, 1)).permute(0, 2, 1)
        if z_e is not None and torch.norm(z_e).item() > 1e-6:
            x_e = self.dec_e(z_e.permute(0, 2, 1)).permute(0, 2, 1)
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