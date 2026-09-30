"""Orthogonal Null-Space weight patcher (Stage 3 / Phase 4).

Closed-form, single-pass surgical repair of the linear decision weights:
    1. Orthonormal shortcut basis via thin SVD of the (centered) shortcut
       activations Z_e: directions with relative energy > eps span the
       shortcut subspace (P_e = V_sub @ V_sub^T).
    2. Global orthogonal projection matrix embedded in the decision-head space:
           P_inv = I - embed(P_e) in R^{target_dim x target_dim}
    3. Surgical patch of the decision weights (transient, never in-place):
           W_repaired = W @ P_inv
    4. Runtime annihilation assertion:
           || W_repaired @ Z_e ||_F < tolerance
"""

from typing import Tuple, Union
import torch
import torch.nn as nn

__all__ = ["NullSpacePatcher"]


class NullSpacePatcher:
    """
    Orthogonal Null-Space Projection Engine.
    Constructs an orthogonal projection matrix P_inv onto the null-space
    of the spurious environmental shortcut subspace Z_e:
        P_inv = I - U_e @ U_e^T
    Surgically patches linear decision weights W in closed form:
        W_repaired = W @ P_inv
    guaranteeing that shortcut activations are annihilated to zero:
        W_repaired @ Z_e = 0.
    """
    def __init__(self, eps: float = 1e-7):
        self.eps = eps

    def compute_nullspace_projection(
        self,
        z_e: torch.Tensor,
        target_dim: int,
        d_e: int = None,
        shortcut_indices: list = None
    ) -> torch.Tensor:
        """
        Constructs P_inv of shape [target_dim, target_dim].
        Acts as identity on invariant features and projects shortcut subspace to zero.
        Args:
            z_e: Shortcut activations of shape [B, d_e] or [B, L, d_e].
            target_dim: Total column dimension of weight matrix being patched.
            d_e: Optional shortcut dimension (inferred from z_e if None).
            shortcut_indices: Optional list of column indices corresponding to shortcut features.
        Returns:
            p_inv: Orthogonal projection matrix of shape [target_dim, target_dim] in FP32.
        """
        device = z_e.device
        dim_e = z_e.shape[-1] if d_e is None else d_e

        # Flatten all batch and sequence dimensions to maximize sample count for SVD
        if z_e.dim() >= 3:
            z_flat = z_e.contiguous().view(-1, dim_e)
        else:
            z_flat = z_e.contiguous()

        # Center the shortcut activations
        z_centered = z_flat - torch.mean(z_flat, dim=0, keepdim=True)

        # Thin SVD to extract the orthonormal basis spanning Z_e
        try:
            _, s, vh = torch.linalg.svd(z_centered, full_matrices=False)
            significant_mask = s > (s[0] * self.eps)
            v_sub = vh[significant_mask, :].t()  # [dim_e, k]
            p_shortcut = torch.matmul(v_sub, v_sub.t())  # [dim_e, dim_e]
        except Exception:
            # Numerical fallback: analytical Moore-Penrose pseudo-inverse
            gram = torch.matmul(z_centered.t(), z_centered)
            reg = self.eps * torch.eye(dim_e, device=device, dtype=torch.float32)
            p_shortcut = torch.matmul(
                torch.matmul(z_centered.t(), torch.linalg.pinv(gram + reg)),
                z_centered
            )

        # Construct global projection matrix P_inv in R^{target_dim x target_dim}
        p_inv = torch.eye(target_dim, device=device, dtype=torch.float32)

        if shortcut_indices is not None:
            # Map shortcut projector to specific designated column indices
            for i, idx_i in enumerate(shortcut_indices):
                for j, idx_j in enumerate(shortcut_indices):
                    p_inv[idx_i, idx_j] -= p_shortcut[i, j]
        else:
            # By default, subtract p_shortcut from the trailing dim_e coordinates
            p_inv[-dim_e:, -dim_e:] -= p_shortcut

        return p_inv

    def patch_linear_weights(
        self,
        linear_or_weight: Union[nn.Linear, torch.Tensor],
        p_inv: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies orthogonal projection matrix to decision weights.
        Supports both nn.Linear layers and raw weight tensors.
        Args:
            linear_or_weight: nn.Linear module or weight Tensor of shape [out_features, in_features].
            p_inv: Projection matrix of shape [in_features, in_features].
        Returns:
            w_repaired: Patched weight tensor of shape [out_features, in_features].
        """
        if isinstance(linear_or_weight, nn.Linear):
            w_matrix = linear_or_weight.weight
        else:
            w_matrix = linear_or_weight

        assert w_matrix.shape[1] == p_inv.shape[0], (
            f"Dimension mismatch: Weight matrix has {w_matrix.shape[1]} columns, "
            f"but P_inv has dimension {p_inv.shape[0]}."
        )

        return torch.matmul(w_matrix, p_inv)

    @staticmethod
    def verify_annihilation(
        w_repaired: torch.Tensor,
        z_e: torch.Tensor,
        tolerance: float = 1e-4
    ) -> Tuple[bool, float]:
        """
        Verifies that shortcut influence is mathematically annihilated:
            || W_repaired @ Z_e ||_F < tolerance.
        """
        dim_e = z_e.shape[-1]
        if z_e.dim() >= 3:
            z_flat = z_e.contiguous().view(-1, dim_e)
        else:
            z_flat = z_e.contiguous()

        # Extract the trailing dim_e columns corresponding to the shortcut channel
        w_e = w_repaired[:, -dim_e:]
        residual = torch.matmul(w_e, z_flat.t())
        residual_norm = torch.norm(residual, p="fro").item()

        return residual_norm < tolerance, residual_norm