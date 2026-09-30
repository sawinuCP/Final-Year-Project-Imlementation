"""Standard Empirical Risk Minimization (ERM) PatchTST Baseline.

Operates directly in the raw observation space [B, L, D] -> [B, H, D].
Channel-independent attention with no latent disentanglement or invariant regularization.
Reference: Nie et al. (ICLR 2023) "A Time Series is Worth 64 Words".
"""

import torch
import torch.nn as nn

__all__ = ["ERMPatchTST"]


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len: int, stride: int, d_model: int):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B * D, L] -> unfold -> [B * D, num_patches, patch_len] -> [B * D, num_patches, d_model]
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return self.proj(patches)


class ERMPatchTST(nn.Module):
    """
    Standard Observation-Space PatchTST trained via ERM (MSE loss).
    """
    def __init__(
        self,
        in_features: int = 7,
        seq_len: int = 96,
        pred_len: int = 96,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1
    ):
        super(ERMPatchTST, self).__init__()
        self.in_features = in_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        self.num_patches = (seq_len - patch_len) // stride + 1
        self.patch_embedding = PatchEmbedding(patch_len, stride, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

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

        self.head_in_dim = self.num_patches * d_model
        self.dropout = nn.Dropout(dropout)
        self.linear_head = nn.Linear(self.head_in_dim, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw input observations [B, L, D]
        Returns:
            Forecasted observations [B, pred_len, D]
        """
        b, l, d = x.shape
        assert l == self.seq_len and d == self.in_features, f"Input shape mismatch: expected [B, {self.seq_len}, {self.in_features}], got {x.shape}"

        # Channel Independence: [B, L, D] -> [B, D, L] -> [B * D, L]
        x_trans = x.permute(0, 2, 1).contiguous().view(b * d, l)

        # Patch projection + positional embedding: [B * D, num_patches, d_model]
        enc_in = self.patch_embedding(x_trans) + self.pos_embedding

        # Transformer encoding: [B * D, num_patches, d_model]
        enc_out = self.transformer_encoder(enc_in)

        # Flatten patches: [B * D, head_in_dim]
        h_flat = enc_out.reshape(b * d, self.head_in_dim)
        out_flat = self.linear_head(self.dropout(h_flat))  # [B * D, pred_len]

        # Reshape to [B, pred_len, D]
        return out_flat.view(b, d, self.pred_len).permute(0, 2, 1).contiguous()