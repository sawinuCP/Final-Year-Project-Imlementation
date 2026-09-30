"""Channel-independent PatchTST forecaster over the Z_inv manifold (Phase 3).

Adapted PatchTST (Nie et al., ICLR 2023), as-built:
    input   Z_inv in R^{B x L x d_inv}
    output  Z_hat in R^{B x H x d_inv}
    patching: patch_len P = 16, stride S = 8  ->  num_patches = 11 (L = 96)
    channel independence: each latent dimension is embedded and attended
    independently through shared Transformer weights (no cross-channel
    pollution), matching the factorized invariant manifold.

The backbone is explicitly decoupled into ``extract_features`` (F_backbone)
and ``forward_head`` (linear decision layer W with dropout) to support
last-layer activation caching for MC-Dropout sampling.
"""

import torch
import torch.nn as nn

__all__ = ["PatchEmbedding", "LatentPatchTST"]


class PatchEmbedding(nn.Module):
    """
    Splits 1D sequence into overlapping patches and projects them to d_model.
    """
    def __init__(self, patch_len: int, stride: int, d_model: int):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B * d_inv, L]
        Returns:
            patches: Tensor of shape [B * d_inv, num_patches, d_model]
        """
        # Unfold sequence along time dimension: [B * d_inv, num_patches, patch_len]
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return self.proj(patches)

class LatentPatchTST(nn.Module):
    """
    Channel-Independent PatchTST for Latent-Space Forecasting.
    Operates strictly on Z_inv [B, L, d_inv].
    Each latent dimension is processed independently through shared Transformer weights.
    """
    def __init__(self, seq_len: int = 96, pred_len: int = 96, d_inv: int = 8,
                 patch_len: int = 16, stride: int = 8, d_model: int = 64,
                 n_heads: int = 4, e_layers: int = 2, d_ff: int = 128,
                 dropout: float = 0.1):
        super(LatentPatchTST, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_inv = d_inv
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        # Calculate number of patches
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.patch_embedding = PatchEmbedding(patch_len, stride, d_model)

        # Learnable Positional Embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        # Channel-independent Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # Flattened representation dimension before the linear head
        self.head_in_dim = self.num_patches * d_model

        # Final linear forecasting decision head (W)
        self.dropout = nn.Dropout(dropout)
        self.linear_head = nn.Linear(self.head_in_dim, pred_len)

    def extract_features(self, z_inv: torch.Tensor) -> torch.Tensor:
        """
        Extracts deep latent features prior to the linear decision layer.
        Args:
            z_inv: [B, L, d_inv]
        Returns:
            h_cached: [B, d_inv, head_in_dim]
        """
        b, l, d = z_inv.shape
        assert l == self.seq_len, f"Sequence length mismatch. Expected {self.seq_len}, got {l}"
        assert d == self.d_inv, f"Latent dimension mismatch. Expected {self.d_inv}, got {d}"

        # Channel independence: reshape so each channel is an independent sample
        # [B, L, d_inv] -> [B, d_inv, L] -> [B * d_inv, L]
        z_trans = z_inv.permute(0, 2, 1).contiguous().view(b * d, l)

        # Patch and embed: [B * d_inv, num_patches, d_model]
        enc_in = self.patch_embedding(z_trans) + self.pos_embedding

        # Transformer representation: [B * d_inv, num_patches, d_model]
        enc_out = self.transformer_encoder(enc_in)

        # Flatten patches: [B * d_inv, num_patches * d_model]
        h_flat = enc_out.reshape(b * d, self.head_in_dim)

        # Reshape back to [B, d_inv, head_in_dim]
        return h_flat.view(b, d, self.head_in_dim)

    def forward_head(self, h_cached: torch.Tensor) -> torch.Tensor:
        """
        Projects cached features to future forecast horizon using linear head W.
        Args:
            h_cached: [B, d_inv, head_in_dim]
        Returns:
            z_hat: [B, pred_len, d_inv]
        """
        b, d, feat_dim = h_cached.shape
        h_flat = h_cached.view(b * d, feat_dim)

        # Apply dropout and linear head projection
        h_drop = self.dropout(h_flat)
        out_flat = self.linear_head(h_drop)  # [B * d_inv, pred_len]

        # Reshape back to [B, pred_len, d_inv]
        out = out_flat.view(b, d, self.pred_len).permute(0, 2, 1).contiguous()
        return out

    def forward(self, z_inv: torch.Tensor) -> torch.Tensor:
        """
        Standard single forward pass.
        """
        h = self.extract_features(z_inv)
        return self.forward_head(h)

