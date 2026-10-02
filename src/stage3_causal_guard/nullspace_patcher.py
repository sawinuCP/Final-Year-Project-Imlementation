"""Direct Latent Null-Space Projection Engine."""

import torch

__all__ = ["NullSpacePatcher"]


class NullSpacePatcher:
    def __init__(self, eps: float = 1e-7):
        self.eps = eps

    def compute_latent_projection(self, z_e: torch.Tensor, d_inv: int) -> torch.Tensor:
        """
        Constructs P_inv in R^{d_inv x d_inv} to nullify components correlated with Z_e.
        """
        device = z_e.device
        b, l, d_e = z_e.shape
        z_flat = z_e.contiguous().view(-1, d_e)
        z_c = z_flat - z_flat.mean(dim=0, keepdim=True)

        # Extract dominant singular vector of the shortcut
        _, s, vh = torch.linalg.svd(z_c, full_matrices=False)
        v_shortcut = vh[0:1, :] # [1, d_e]

        # Construct projection matrix in R^{d_inv x d_inv}
        p_inv = torch.eye(d_inv, device=device, dtype=torch.float32)
        # Suppress the corresponding coupled dimensions
        for i in range(min(d_inv, d_e)):
            p_inv[i, i] = max(0.0, 1.0 - torch.norm(v_shortcut[:, i]).item())

        return p_inv

    def project_latent_trajectory(self, z_pred: torch.Tensor, p_inv: torch.Tensor) -> torch.Tensor:
        """Applies P_inv directly to predicted latent trajectory: Z_repaired = Z_pred @ P_inv."""
        return torch.matmul(z_pred, p_inv)