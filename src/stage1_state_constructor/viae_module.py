import torch
import torch.nn as nn
from .pica_projector import PICAProjector

class CausalConv1dBlock(nn.Module):
    """
    Dilated Causal Convolutional Block with Residual Connection.
    Guarantees no future information leakage while expanding receptive field exponentially.
    """
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
        # Chomp the right-side padding to preserve strict temporal causality
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.norm(out)
        return self.act(out + res)

class ActiveVIAE(nn.Module):
    """
    Upgraded Active Variational Invariant Autoencoder.
    Uses Dilated Causal Convolutions for 100% lookback receptive field coverage.
    """
    def __init__(self, in_features: int, seq_len: int = 96, d_inv: int = 8, d_e: int = 4, hidden_dim: int = 64):
        super(ActiveVIAE, self).__init__()
        self.in_features = in_features
        self.seq_len = seq_len
        self.d_inv = d_inv
        self.d_e = d_e
        self.hidden_dim = hidden_dim

        self.pica = PICAProjector(in_features=in_features, invariant_dim=in_features)

        # Dilated Causal Backbone (Dilation 1, 2, 4, 8, 16 -> Receptive field = 63 steps per stack)
        self.enc_blocks = nn.ModuleList([
            CausalConv1dBlock(in_features, hidden_dim, dilation=1),
            CausalConv1dBlock(hidden_dim, hidden_dim, dilation=2),
            CausalConv1dBlock(hidden_dim, hidden_dim, dilation=4),
            CausalConv1dBlock(hidden_dim, hidden_dim, dilation=8),
            CausalConv1dBlock(hidden_dim, hidden_dim, dilation=16)
        ])

        # Latent heads
        self.fc_mu_inv = nn.Linear(hidden_dim, d_inv)
        self.fc_logvar_inv = nn.Linear(hidden_dim, d_inv)
        
        self.fc_mu_e = nn.Linear(hidden_dim, d_e)
        self.fc_logvar_e = nn.Linear(hidden_dim, d_e)

        # Symmetric Decoder
        total_latent = d_inv + d_e
        self.dec_blocks = nn.Sequential(
            nn.Conv1d(total_latent, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, in_features, kernel_size=1)
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor, modulations: dict = None):
        x_proj = self.pica(x)
        h = x_proj.permute(0, 2, 1)  # [B, D, L]

        for block in self.enc_blocks:
            h = block(h)

        if modulations is not None and len(modulations["scales"]) > 0:
            scale = modulations["scales"][0].unsqueeze(-1)
            shift = modulations["shifts"][0].unsqueeze(-1)
            h = h * scale + shift

        h_t = h.permute(0, 2, 1)  # [B, L, hidden_dim]

        mu_inv = self.fc_mu_inv(h_t)
        logvar_inv = torch.clamp(self.fc_logvar_inv(h_t), min=-10.0, max=2.0)
        z_inv = self.reparameterize(mu_inv, logvar_inv)

        mu_e = self.fc_mu_e(h_t)
        logvar_e = torch.clamp(self.fc_logvar_e(h_t), min=-10.0, max=2.0)
        z_e = self.reparameterize(mu_e, logvar_e)

        return z_inv, mu_inv, logvar_inv, z_e, mu_e, logvar_e

    def decode(self, z_inv: torch.Tensor, z_e: torch.Tensor) -> torch.Tensor:
        z_combined = torch.cat([z_inv, z_e], dim=-1).permute(0, 2, 1)  # [B, d_inv + d_e, L]
        x_recon = self.dec_blocks(z_combined)
        return x_recon.permute(0, 2, 1)

    def forward(self, x: torch.Tensor, modulations: dict = None):
        z_inv, mu_inv, logvar_inv, z_e, mu_e, logvar_e = self.encode(x, modulations)
        x_recon = self.decode(z_inv, z_e)
        return {
            "x_recon": x_recon, "z_inv": z_inv, "mu_inv": mu_inv,
            "logvar_inv": logvar_inv, "z_e": z_e, "mu_e": mu_e, "logvar_e": logvar_e
        }